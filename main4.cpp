// =================================================================
// main4.cpp — MLP configurable sobre MNIST.
//
// Permite elegir por linea de comandos:
//   - device  : cpu | cuda
//   - dtype   : f32  | f64 | i32 | i64
//   - capas   : lista de tamanos, p.ej. 784,128,64,10
//   - y otros hiperparametros (epocas, batch, lr, limites, paths)
//
// Uso:
//   ./train_mnist.out [--device cuda] [--dtype f32] [--layers 784,128,64,10] ...
//   (o simplemente ./train.sh con las mismas opciones)
// =================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <stdexcept>

#include "lib/tensor.h"
#include "lib/ops.h"
#include "lib/linear.h"
#include "lib/loss.h"
#include "lib/optim.h"
#include "lib/dataset.h"

#include <cuda_runtime.h>

// ============================================================
// Utilidades
// ============================================================

static const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32: return "F32";
        case DType::F64: return "F64";
        case DType::I32: return "I32";
        case DType::I64: return "I64";
    }
    return "?";
}

static void print_usage() {
    std::cout
        << "Uso: train_mnist.out [opciones]\n\n"
        << "  --device cpu|cuda        Device de entrenamiento       (def: cuda)\n"
        << "  --dtype  f32|f64|i32|i64 Tipo de dato                   (def: f32)\n"
        << "  --layers 784,128,64,10   Arquitectura del MLP           (def: 784,128,64,10)\n"
        << "  --epochs N               Numero de epocas               (def: 15)\n"
        << "  --batch N                Tamano del minilote            (def: 64)\n"
        << "  --lr F                   Learning rate                  (def: 0.08)\n"
        << "  --limit N                Limitar muestras de train      (def: -1, todas)\n"
        << "  --eval-batch N           Batch de evaluacion            (def: 128)\n"
        << "  --train PATH             CSV de entrenamiento\n"
        << "  --test PATH              CSV de test\n"
        << "  --help                   Muestra esta ayuda\n";
}

// Convierte un tensor CPU/F32 (los batches de MNIST) al device/dtype pedidos.
// Para dtypes enteros los valores [0,1] de MNIST se truncan a 0/1: el
// entrenamiento con enteros es degenerado (no converge), pero estos runs
// permiten verificar que todo el pipeline soporta los 4 dtypes.
static Tensor to_device(const Tensor& src, Device device, DType dtype) {
    Tensor dst(src.shape(), dtype, device);
    const uint64_t n = src.numel();
    const float* s = src.data_ptr<float>();

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        std::vector<scalar_t> tmp(n);
        for (uint64_t i = 0; i < n; ++i) tmp[i] = static_cast<scalar_t>(s[i]);
        if (device == Device::CPU) {
            std::memcpy(dst.data(), tmp.data(), n * sizeof(scalar_t));
        } else {
            cudaMemcpy(dst.data(), tmp.data(), n * sizeof(scalar_t), cudaMemcpyHostToDevice);
        }
    });
    return dst;
}

// Lee un escalar (loss) desde CPU o GPU, para los 4 dtypes.
static double read_scalar(const Tensor& t) {
    if (t.device() == Device::CUDA) {
        if (t.dtype() == DType::F32) { float v; cudaMemcpy(&v, t.data(), sizeof(float), cudaMemcpyDeviceToHost); return (double)v; }
        if (t.dtype() == DType::F64) { double v; cudaMemcpy(&v, t.data(), sizeof(double), cudaMemcpyDeviceToHost); return v; }
        if (t.dtype() == DType::I32) { int32_t v; cudaMemcpy(&v, t.data(), sizeof(int32_t), cudaMemcpyDeviceToHost); return (double)v; }
        int64_t v; cudaMemcpy(&v, t.data(), sizeof(int64_t), cudaMemcpyDeviceToHost); return (double)v;
    }
    if (t.dtype() == DType::F32) return (double)t.data_ptr<float>()[0];
    if (t.dtype() == DType::F64) return t.data_ptr<double>()[0];
    if (t.dtype() == DType::I32) return (double)t.data_ptr<int32_t>()[0];
    return (double)t.data_ptr<int64_t>()[0];
}

// Descarga un tensor (CPU o GPU, cualquiera de los 4 dtypes) a un vector<float>.
static std::vector<float> to_host_float(const Tensor& t) {
    std::vector<float> out(t.numel());

    if (t.device() == Device::CUDA) {
        std::vector<char> buf(t.numel() * t.get_element_size(t.dtype()));
        cudaMemcpy(buf.data(), t.data(), buf.size(), cudaMemcpyDeviceToHost);
        DISPATCH_ALL_TYPES(t.dtype(), scalar_t, {
            const scalar_t* p = reinterpret_cast<const scalar_t*>(buf.data());
            for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(p[i]);
        });
    } else {
        DISPATCH_ALL_TYPES(t.dtype(), scalar_t, {
            const scalar_t* p = t.data_ptr<scalar_t>();
            for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(p[i]);
        });
    }
    return out;
}

// ============================================================
// MLP generico: Linear + ReLU por capa oculta y una capa SoftMax
// propia al final. El forward devuelve probabilidades; el backward
// arranca con el Jacobiano del SoftMax (dL/dp -> dL/dz).
// ============================================================
struct MLP {
    std::vector<LinearLayer> linears;
    std::vector<ReLU> relus;
    SoftMax softmax_layer;
    Device device;
    DType dtype;

    MLP(const std::vector<int64_t>& sizes, Device dev, DType dt)
        : device(dev), dtype(dt)
    {
        // reserve para que los punteros que registra el optimizador sean estables
        linears.reserve(sizes.size() - 1);
        for (size_t i = 0; i + 1 < sizes.size(); ++i) {
            linears.emplace_back(sizes[i], sizes[i + 1], dev, dt);
        }
        // Un ReLU tras cada capa oculta (la ultima capa no lleva activacion)
        relus.resize(linears.size() - 1);
    }

    Tensor forward(const Tensor& x) {
        Tensor h = x;
        for (size_t i = 0; i < linears.size(); ++i) {
            h = linears[i].forward(h);
            if (i + 1 < linears.size()) h = relus[i].forward(h);
        }
        return softmax_layer.forward(h);  // logits -> probabilidades
    }

    void backward(const Tensor& d_output) {
        // d_output = dL/dp (gradiente respecto a las probabilidades)
        Tensor d = softmax_layer.backward(d_output);  // -> dL/dz (logits)
        d = linears.back().backward(d);
        for (int64_t i = static_cast<int64_t>(linears.size()) - 2; i >= 0; --i) {
            d = relus[i].backward(d);
            d = linears[i].backward(d);
        }
    }
};

// ============================================================
// Evaluacion de accuracy
// ============================================================
static float evaluate_accuracy(MLP& model, const MNISTDataset& dataset,
                               Device device, DType dtype, int64_t eval_batch) {
    if (dataset.count < eval_batch) return 0.0f;

    int64_t total_correct = 0;
    const int64_t num_batches = dataset.count / eval_batch;

    for (int64_t b = 0; b < num_batches; ++b) {
        Tensor bx_cpu, by_cpu;
        dataset.get_batch(b * eval_batch, eval_batch, bx_cpu, by_cpu);

        Tensor bx = to_device(bx_cpu, device, dtype);
        Tensor probs = model.forward(bx);  // el MLP termina en la capa SoftMax

        std::vector<float> hp = to_host_float(probs);
        const float* gt = by_cpu.data_ptr<float>();

        for (int64_t i = 0; i < eval_batch; ++i) {
            int pred = 0, actual = 0;
            float best = -1.0f;
            for (int c = 0; c < 10; ++c) {
                float p = hp[i * 10 + c];
                if (p > best) { best = p; pred = c; }
                if (gt[i * 10 + c] == 1.0f) actual = c;
            }
            if (pred == actual) total_correct++;
        }
    }

    return 100.0f * static_cast<float>(total_correct) / static_cast<float>(num_batches * eval_batch);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    // ---------------- Configuracion por defecto ----------------
    Device device = Device::CUDA;
    DType dtype = DType::F32;
    std::vector<int64_t> sizes = {784, 128, 64, 10};
    int epochs = 15;
    int64_t batch_size = 64;
    float lr = 0.08f;
    int64_t limit = -1;
    int64_t eval_batch = 128;
    std::string train_path = "data/mnist-main/mnist_train.csv/mnist_train.csv";
    std::string test_path  = "data/mnist-main/mnist_test.csv/mnist_test.csv";

    // ---------------- Parseo de argumentos ----------------
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("Falta el valor de ") + name);
                return argv[++i];
            };

            if (arg == "--device") {
                std::string v = next("--device");
                if (v == "cpu") device = Device::CPU;
                else if (v == "cuda") device = Device::CUDA;
                else throw std::runtime_error("--device debe ser cpu o cuda");
            } else if (arg == "--dtype") {
                std::string v = next("--dtype");
                if (v == "f32") dtype = DType::F32;
                else if (v == "f64") dtype = DType::F64;
                else if (v == "i32") dtype = DType::I32;
                else if (v == "i64") dtype = DType::I64;
                else throw std::runtime_error("--dtype debe ser f32, f64, i32 o i64");
            } else if (arg == "--layers") {
                sizes.clear();
                std::stringstream ss(next("--layers"));
                std::string tok;
                while (std::getline(ss, tok, ',')) sizes.push_back(std::stoll(tok));
            } else if (arg == "--epochs") {
                epochs = std::stoi(next("--epochs"));
            } else if (arg == "--batch") {
                batch_size = std::stoll(next("--batch"));
            } else if (arg == "--lr") {
                lr = std::stof(next("--lr"));
            } else if (arg == "--limit") {
                limit = std::stoll(next("--limit"));
            } else if (arg == "--eval-batch") {
                eval_batch = std::stoll(next("--eval-batch"));
            } else if (arg == "--train") {
                train_path = next("--train");
            } else if (arg == "--test") {
                test_path = next("--test");
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else {
                std::cerr << "[ERROR] Argumento desconocido: " << arg << "\n\n";
                print_usage();
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    // ---------------- Validaciones ----------------
    if (sizes.size() < 2) {
        std::cerr << "[ERROR] --layers necesita al menos 2 tamanos (entrada y salida)\n";
        return 1;
    }
    if (sizes.front() != 784) {
        std::cerr << "[ERROR] La primera capa debe ser 784 (pixeles de MNIST)\n";
        return 1;
    }
    if (sizes.back() != 10) {
        std::cerr << "[ERROR] La ultima capa debe ser 10 (clases de MNIST)\n";
        return 1;
    }
    if (batch_size <= 0 || epochs <= 0) {
        std::cerr << "[ERROR] --batch y --epochs deben ser positivos\n";
        return 1;
    }

    // ---------------- Comprobacion de GPU ----------------
    std::string gpu_name;
    if (device == Device::CUDA) {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0) {
            std::cerr << "[ERROR] No hay GPUs CUDA disponibles. Usa --device cpu\n";
            return 1;
        }
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        gpu_name = prop.name;
    }

    // ---------------- Banner ----------------
    std::cout << "==========================================\n";
    std::cout << "   MLP CONFIGURABLE - MNIST (main4)       \n";
    std::cout << "==========================================\n";
    std::cout << " Device:     " << (device == Device::CUDA ? "CUDA" : "CPU");
    if (device == Device::CUDA) std::cout << " (" << gpu_name << ")";
    std::cout << "\n";
    std::cout << " Dtype:      " << dtype_name(dtype) << "\n";
    std::cout << " Capas:      ";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i] << (i + 1 < sizes.size() ? " -> " : "");
    }
    std::cout << "\n";
    std::cout << " Epocas:     " << epochs << "\n";
    std::cout << " Batch:      " << batch_size << "\n";
    std::cout << " LR:         " << lr << "\n";
    std::cout << "==========================================\n\n";

    if (dtype == DType::I32 || dtype == DType::I64) {
        std::cout << "[NOTA] Dtype entero: los pesos (He ~0.05), el learning rate y los\n"
                     "       pixeles [0,1] se truncan a 0, asi que el entrenamiento no\n"
                     "       converge. Este run verifica que el pipeline soporta todos\n"
                     "       los dtypes de principio a fin.\n\n";
    }

    // ---------------- Dataset ----------------
    MNISTDataset train_data = MNISTDataset::load_csv(train_path, limit);

    if (train_data.count < batch_size) {
        std::cerr << "[ERROR] Menos muestras (" << train_data.count << ") que el batch (" << batch_size << ")\n";
        return 1;
    }
    const int64_t num_batches = train_data.count / batch_size;
    std::cout << "[INFO] Muestras de entrenamiento: " << train_data.count
              << " | Batches por epoca: " << num_batches << "\n";

    // ---------------- Modelo + optimizador ----------------
    MLP model(sizes, device, dtype);
    SGD optimizer(lr);
    for (auto& layer : model.linears) {
        optimizer.register_layer(layer);
    }

    // ---------------- Entrenamiento ----------------
    std::cout << "\n==========================================\n";
    std::cout << "            ENTRENAMIENTO                 \n";
    std::cout << "==========================================\n";

    const auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        const auto t_epoch = std::chrono::steady_clock::now();
        double epoch_loss = 0.0;

        for (int64_t b = 0; b < num_batches; ++b) {
            Tensor bx_cpu, by_cpu;
            train_data.get_batch(b * batch_size, batch_size, bx_cpu, by_cpu);

            Tensor bx = to_device(bx_cpu, device, dtype);
            Tensor by = to_device(by_cpu, device, dtype);

            // 1. Limpieza de gradientes
            optimizer.zero_grad();

            // 2. Forward (el MLP termina en la capa SoftMax -> probabilidades)
            Tensor probs = model.forward(bx);

            // 3. Loss + gradiente respecto a las probabilidades (dL/dp)
            LossResult res = cross_entropy_probs(probs, by, device, dtype);

            // 4. Backward
            model.backward(res.grad);

            // 5. Actualizacion W = W - lr * dW
            optimizer.step();

            epoch_loss += read_scalar(res.loss);
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t_end - t_epoch).count();

        std::cout << "Epoca [" << std::setw(2) << epoch << "/" << epochs << "]"
                  << " | Loss Promedio: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches)
                  << " | " << std::setprecision(2) << secs << "s\n";
    }

    const double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    // ---------------- Evaluacion ----------------
    std::cout << "\n==========================================\n";
    std::cout << "            EVALUACION FINAL              \n";
    std::cout << "==========================================\n";

    float train_acc = evaluate_accuracy(model, train_data, device, dtype, eval_batch);
    std::cout << "Precision en Entrenamiento: " << std::fixed << std::setprecision(2) << train_acc << "%\n";

    try {
        MNISTDataset test_data = MNISTDataset::load_csv(test_path);
        float test_acc = evaluate_accuracy(model, test_data, device, dtype, eval_batch);
        std::cout << "Precision en Test Set:      " << std::fixed << std::setprecision(2) << test_acc << "%\n";
    } catch (const std::exception& e) {
        std::cout << "[INFO] No se pudo cargar test set (" << e.what() << ")\n";
    }

    std::cout << "==========================================\n";
    std::cout << "Tiempo total de entrenamiento: " << std::setprecision(2) << total_secs << "s\n";
    std::cout << "==========================================\n";

    return 0;
}
