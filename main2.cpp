#include <iostream>
#include <cassert>
#include <cmath>

#include "lib/tensor.h"
#include "lib/ops.h"
#include "lib/linear.h"
#include "lib/loss.h"

// ============================================================
// Definición del Perceptrón Multicapa para MNIST (784 -> 128 -> 64 -> 10)
// ============================================================
class MNISTNet {
public:
    LinearLayer fc1;
    LinearLayer fc2;
    LinearLayer fc3;

    MNISTNet(Device device = Device::CPU, DType dtype = DType::F32)
        : fc1(784, 128, device, dtype),
          fc2(128, 64, device, dtype),
          fc3(64, 10, device, dtype) {}

    Tensor forward(const Tensor& x) {
        // x: [Batch, 784]
        Tensor h1 = fc1.forward(x);
        Tensor a1 = relu(h1);

        Tensor h2 = fc2.forward(a1);
        Tensor a2 = relu(h2);

        Tensor logits = fc3.forward(a2);
        Tensor probs  = softmax(logits); // [Batch, 10]

        return probs;
    }
};

int main() {
    std::cout << "==========================================\n";
    std::cout << "       TEST - RED NEURONAL PARA MNIST     \n";
    std::cout << "==========================================\n\n";

    // 1. Instanciar la red neuronal
    MNISTNet model(Device::CPU, DType::F32);
    std::cout << "[OK] Modelo MNISTNet instanciado (784 -> 128 -> 64 -> 10)\n";

    // 2. Simular un lote de entrada (Batch = 2 imágenes de 28x28 aplanadas a 784)
    const int64_t batch_size = 2;
    const int64_t in_features = 784;
    const int64_t num_classes = 10;

    // Generamos entrada simulada normalizada [0, 1]
    Tensor input = Tensor::randn({batch_size, in_features}, 0.5f, 0.2f, DType::F32, Device::CPU);
    std::cout << "[OK] Batch de entrada creado con dimensiones: [" 
              << input.shape()[0] << ", " << input.shape()[1] << "]\n";

    // 3. Ejecutar el Forward Pass completo
    Tensor probs = model.forward(input);
    std::cout << "[OK] Forward pass completado con exito\n";

    // 4. Validaciones de dimensiones y número de elementos
    assert(probs.shape().size() == 2);
    assert(probs.shape()[0] == batch_size);
    assert(probs.shape()[1] == num_classes);
    assert(probs.numel() == static_cast<uint64_t>(batch_size * num_classes));
    std::cout << "[OK] Output shape correcta: [" << probs.shape()[0] << ", " << probs.shape()[1] << "]\n";

    // 5. Verificar propiedades de Softmax (la suma de cada fila debe ser ~1.0)
    const float* probs_ptr = probs.data_ptr<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        int best_digit = 0;
        float max_prob = -1.0f;

        for (int64_t c = 0; c < num_classes; ++c) {
            float p = probs_ptr[b * num_classes + c];
            sum += p;
            if (p > max_prob) {
                max_prob = p;
                best_digit = c;
            }
        }

        std::cout << "Muestra #" << b 
                  << " -> Prediccion inicial (sin entrenar): digito " << best_digit 
                  << " con probabilidad: " << max_prob * 100.0f << "%\n";

        // La suma de probabilidades debe estar extremadamente cerca de 1.0f
        assert(std::fabs(sum - 1.0f) < 1e-4f);
    }
    std::cout << "[OK] Validacion Softmax: las probabilidades de cada muestra suman 1.0\n";

    // 6. Visualizar las probabilidades de la primera muestra
    std::cout << "\nProbabilidades del primer digito:\n" << probs << "\n";


    // ============================================================
    // 7. Prueba de Cross Entropy Loss
    // ============================================================
    std::cout << "\n--- TEST: Calculo de Cross Entropy Loss ---\n";

    // Creamos ground truth en formato One-Hot con la misma shape que probs: [2, 10]
    // Supongamos que:
    //  - Muestra 0 es el dígito 3  -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    //  - Muestra 1 es el dígito 7  -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    Tensor gt = Tensor::zeros({2, 10}, DType::F32, Device::CPU);
    float* gt_ptr = gt.data_ptr<float>();

    // Muestra 0: índice 3
    gt_ptr[0 * 10 + 3] = 1.0f;
    // Muestra 1: índice 7
    gt_ptr[1 * 10 + 7] = 1.0f;

    std::cout << "Ground Truth One-Hot:\n" << gt << "\n";

    // Calculamos la pérdida
    Tensor loss = cross_entropy(probs, gt, Device::CPU, DType::F32);

    std::cout << "[OK] Perdida calculada con exito\n";
    std::cout << "Loss tensor: " << loss << "\n";
    std::cout << "Loss scalar: " << loss.data_ptr<float>()[0] << "\n";

    // Con 10 clases no entrenadas (probabilidad inicial ~0.10),
    // la pérdida teórica esperada es -ln(0.1) ≈ 2.302
    float loss_val = loss.data_ptr<float>()[0];
    assert(loss_val > 0.0f);
    assert(!std::isnan(loss_val) && !std::isinf(loss_val));

    std::cout << "\n==========================================\n";
    std::cout << "       TEST DE INFERENCIA EXITOSO         \n";
    std::cout << "==========================================\n";

    return 0;
}