#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

#include "lib/tensor.h"
#include "lib/ops.h"
#include "lib/linear.h"
#include "lib/loss.h"
#include "lib/optim.h"
#include "lib/dataset.h"

inline std::string shape_to_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) s += ", ";
    }
    s += "]";
    return s;
}

// ============================================================
// Definición del Perceptrón Multicapa para MNIST (784 -> 128 -> 64 -> 10)
// ============================================================
class MNISTNet {
public:
    LinearLayer fc1;
    ReLU relu1;
    LinearLayer fc2;
    ReLU relu2;
    LinearLayer fc3;

    MNISTNet(Device device = Device::CPU, DType dtype = DType::F32)
        : fc1(784, 128, device, dtype),
          fc2(128, 64, device, dtype),
          fc3(64, 10, device, dtype) {}

    Tensor forward(const Tensor& x, bool print = false) {
        Tensor h1 = fc1.forward(x);
        Tensor a1 = relu1.forward(h1);
        Tensor h2 = fc2.forward(a1);
        Tensor a2 = relu2.forward(h2);
        Tensor logits = fc3.forward(a2);
        Tensor probs = softmax(logits);

        if (print) {
            std::cout << "\n=== FORWARD PASS (SHAPES) ===\n";
            std::cout << "Input X:         " << shape_to_str(x.shape()) << "\n";          
            std::cout << "fc1 output:      " << shape_to_str(h1.shape()) << "\n";
            std::cout << "relu1 output:    " << shape_to_str(a1.shape()) << "\n";
            std::cout << "fc2 output:      " << shape_to_str(h2.shape()) << "\n";
            std::cout << "relu2 output:    " << shape_to_str(a2.shape()) << "\n";
            std::cout << "fc3 output:      " << shape_to_str(logits.shape()) << "\n";
            std::cout << "softmax output:  " << shape_to_str(probs.shape()) << "\n";
            std::cout << "=============================\n";
        }

        return probs;
    }

    void backward(const Tensor& d_logits, bool print = false) {
        Tensor d_a2 = fc3.backward(d_logits);
        Tensor d_h2 = relu2.backward(d_a2);
        Tensor d_a1 = fc2.backward(d_h2);
        Tensor d_h1 = relu1.backward(d_a1);
        Tensor d_x  = fc1.backward(d_h1);

        if (print) {
            std::cout << "\n=== BACKWARD PASS (SHAPES) ===\n";
            std::cout << "d_logits (dL/dz): " << shape_to_str(d_logits.shape()) << "\n";
            std::cout << "fc3 -> d_weights: " << shape_to_str(fc3.d_weights.shape()) << "\n";
            std::cout << "fc3 -> d_bias:    " << shape_to_str(fc3.d_bias.shape()) << "\n";
            std::cout << "fc3 -> d_in:      " << shape_to_str(d_a2.shape()) << "\n";
            std::cout << "relu2 -> d_in:    " << shape_to_str(d_h2.shape()) << "\n";
            std::cout << "fc2 -> d_weights: " << shape_to_str(fc2.d_weights.shape()) << "\n";
            std::cout << "fc2 -> d_bias:    " << shape_to_str(fc2.d_bias.shape()) << "\n";
            std::cout << "fc2 -> d_in:      " << shape_to_str(d_a1.shape()) << "\n";
            std::cout << "relu1 -> d_in:    " << shape_to_str(d_h1.shape()) << "\n";
            std::cout << "fc1 -> d_weights: " << shape_to_str(fc1.d_weights.shape()) << "\n";
            std::cout << "fc1 -> d_bias:    " << shape_to_str(fc1.d_bias.shape()) << "\n";
            std::cout << "fc1 -> d_in (dX): " << shape_to_str(d_x.shape()) << "\n";
            std::cout << "==============================\n";
        }
    }
};

// ============================================================
// Función de Evaluación: Calcula el Accuracy en %
// ============================================================
float evaluate_accuracy(MNISTNet& model, const MNISTDataset& dataset, int64_t eval_batch_size = 128) {
    int64_t total_correct = 0;
    int64_t num_batches = dataset.count / eval_batch_size;

    for (int64_t b = 0; b < num_batches; ++b) {
        Tensor batch_x, batch_y;
        dataset.get_batch(b * eval_batch_size, eval_batch_size, batch_x, batch_y);

        Tensor probs = model.forward(batch_x, false);

        const float* probs_ptr = probs.data_ptr<float>();
        const float* gt_ptr    = batch_y.data_ptr<float>();

        for (int64_t i = 0; i < eval_batch_size; ++i) {
            int pred_digit = 0;
            float max_prob = -1.0f;
            int actual_digit = 0;

            for (int c = 0; c < 10; ++c) {
                float p = probs_ptr[i * 10 + c];
                if (p > max_prob) {
                    max_prob = p;
                    pred_digit = c;
                }
                if (gt_ptr[i * 10 + c] == 1.0f) {
                    actual_digit = c;
                }
            }

            if (pred_digit == actual_digit) {
                total_correct++;
            }
        }
    }

    int64_t evaluated_samples = num_batches * eval_batch_size;
    return (static_cast<float>(total_correct) / static_cast<float>(evaluated_samples)) * 100.0f;
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "       INICIANDO PIPELINE DE MNIST        \n";
    std::cout << "==========================================\n\n";

    // 1. Instanciar la red neuronal
    MNISTNet model(Device::CPU, DType::F32);

    // 2. Cargar Dataset de Entrenamiento y Test
    // Cambia los paths a la ubicación de tus archivos CSV
    std::string train_path = "data/mnist-main/mnist_train.csv/mnist_train.csv";
    std::string test_path  = "data/mnist-main/mnist_test.csv/mnist_test.csv";

    // Carga de entrenamiento (e.g. 10000 muestras para buen balance entre velocidad y precisión, o -1 para todo el dataset)
    MNISTDataset train_data = MNISTDataset::load_csv(train_path);

    // 3. Configurar optimizador y parámetros
    SGD optimizer(0.08f); // Learning rate ajustado para convergencia estable
    optimizer.register_layer(model.fc1);
    optimizer.register_layer(model.fc2);
    optimizer.register_layer(model.fc3);

    const int64_t batch_size = 64;
    const int epochs = 15;
    const int num_batches = train_data.count / batch_size;

    std::cout << "\n==========================================\n";
    std::cout << "       ENTRENAMIENTO (15 EPOCHS)          \n";
    std::cout << "==========================================\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            Tensor batch_x, batch_y;
            train_data.get_batch(b * batch_size, batch_size, batch_x, batch_y);

            // 1. Limpieza de gradientes
            optimizer.zero_grad();

            // 2. Forward pass
            Tensor probs = model.forward(batch_x, false);

            // 3. Loss + Gradiente de la función de coste
            auto [loss, d_logits] = cross_entropy(probs, batch_y, Device::CPU, DType::F32);

            // 4. Backward pass
            model.backward(d_logits, false);

            // 5. Actualización de parámetros W = W - lr * dW
            optimizer.step();

            epoch_loss += loss.data_ptr<float>()[0];
        }

        float avg_loss = epoch_loss / num_batches;
        std::cout << "Epoca [" << std::setw(2) << epoch << "/" << epochs << "] "
                  << "| Loss Promedio: " << std::fixed << std::setprecision(4) << avg_loss << "\n";
    }

    // ============================================================
    // 4. Evaluación de Precisión (Accuracy)
    // ============================================================
    std::cout << "\n==========================================\n";
    std::cout << "           EVALUACION FINAL               \n";
    std::cout << "==========================================\n";

    float train_acc = evaluate_accuracy(model, train_data);
    std::cout << "Precision en Entrenamiento: " << std::fixed << std::setprecision(2) << train_acc << "%\n";

    try {
        MNISTDataset test_data = MNISTDataset::load_csv(test_path);
        float test_acc = evaluate_accuracy(model, test_data);
        std::cout << "Precision en Test Set:      " << std::fixed << std::setprecision(2) << test_acc << "%\n";
    } catch (const std::exception& e) {
        std::cout << "[INFO] No se pudo cargar test set (" << e.what() << "). Evaluacion completada sobre train.\n";
    }

    std::cout << "==========================================\n";

    return 0;
}