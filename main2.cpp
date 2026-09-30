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

    Tensor input = Tensor::randn({batch_size, in_features}, 0.5f, 0.2f, DType::F32, Device::CPU);
    std::cout << "[OK] Batch de entrada creado con dimensiones: [" 
              << input.shape()[0] << ", " << input.shape()[1] << "]\n";

    // 3. Ejecutar el Forward Pass completo
    Tensor probs = model.forward(input);
    std::cout << "[OK] Forward pass completado con exito\n";

    // 4. Validaciones de dimensiones
    assert(probs.shape().size() == 2);
    assert(probs.shape()[0] == batch_size);
    assert(probs.shape()[1] == num_classes);
    assert(probs.numel() == static_cast<uint64_t>(batch_size * num_classes));
    std::cout << "[OK] Output shape correcta: [" << probs.shape()[0] << ", " << probs.shape()[1] << "]\n";

    // 5. Verificar propiedades de Softmax
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
                  << " -> Prediccion inicial: digito " << best_digit 
                  << " con probabilidad: " << max_prob * 100.0f << "%\n";

        assert(std::fabs(sum - 1.0f) < 1e-4f);
    }
    std::cout << "[OK] Validacion Softmax: probabilidades suman 1.0\n";

    // ============================================================
    // 6. Prueba conjunta de Cross Entropy Loss + Gradiente
    // ============================================================
    std::cout << "\n--- TEST: Cross Entropy Loss y Gradiente (dL/dz) ---\n";

    Tensor gt = Tensor::zeros({batch_size, num_classes}, DType::F32, Device::CPU);
    float* gt_ptr = gt.data_ptr<float>();

    // Muestra 0: dígito 3 | Muestra 1: dígito 7
    gt_ptr[0 * num_classes + 3] = 1.0f;
    gt_ptr[1 * num_classes + 7] = 1.0f;

    // Llamada con structured binding a la función unificada
    auto [loss, d_logits] = cross_entropy(probs, gt, Device::CPU, DType::F32);

    std::cout << "[OK] Perdida y gradientes calculados en una sola pasada\n";

    // Validar Loss (escalar de 1 elemento)
    assert(loss.shape().size() == 1 && loss.shape()[0] == 1);
    float loss_val = loss.data_ptr<float>()[0];
    std::cout << "Loss scalar: " << loss_val << "\n";
    assert(loss_val > 0.0f && !std::isnan(loss_val) && !std::isinf(loss_val));

    // Validar d_logits (mismas dimensiones que probs: [2, 10])
    assert(d_logits.shape().size() == 2);
    assert(d_logits.shape()[0] == batch_size);
    assert(d_logits.shape()[1] == num_classes);
    std::cout << "[OK] Gradiente d_logits con dimensiones correctas: ["
              << d_logits.shape()[0] << ", " << d_logits.shape()[1] << "]\n";

    // La suma de los gradientes de una muestra sobre todas las clases debe ser ~0
    // porque sum(P - Y) = sum(P) - sum(Y) = 1.0 - 1.0 = 0.0
    const float* grad_ptr = d_logits.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
        float grad_sum = 0.0f;
        for (int64_t c = 0; c < num_classes; ++c) {
            grad_sum += grad_ptr[b * num_classes + c];
        }
        assert(std::fabs(grad_sum) < 1e-4f);
    }
    std::cout << "[OK] Propiedad analitica de d_logits verificada: sum(grad) == 0 por muestra\n";

    std::cout << "\nGradiente d_logits generado:\n" << d_logits << "\n";

    std::cout << "==========================================\n";
    std::cout << "       TODOS LOS TESTS PASARON            \n";
    std::cout << "==========================================\n";

    return 0;
}