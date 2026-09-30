#include <iostream>
#include <cassert>

#include "lib/tensor.h"
#include "lib/ops.h"
#include "lib/linear.h"


int main() {

    std::cout << "==========================================\n";
    std::cout << "       TEST - LINEAR LAYER                \n";
    std::cout << "==========================================\n\n";


    // ============================================================
    // 1. Crear LinearLayer
    // ============================================================

    const int64_t in_features = 3;
    const int64_t out_features = 2;

    LinearLayer layer(
        in_features,
        out_features,
        Device::CPU,
        DType::F32
    );

    std::cout << "[OK] LinearLayer creado\n";


    // ============================================================
    // 2. Crear input
    // ============================================================

    // Batch de 2 muestras
    //
    // [1, 2, 3]
    // [4, 5, 6]
    //
    // Shape = [2, 3]

    Tensor input(
        {2, 3},
        DType::F32,
        Device::CPU
    );

    float* input_ptr = static_cast<float*>(input.data());

    input_ptr[0] = 1.0f;
    input_ptr[1] = 2.0f;
    input_ptr[2] = 3.0f;

    input_ptr[3] = 4.0f;
    input_ptr[4] = 5.0f;
    input_ptr[5] = 6.0f;


    std::cout << "[OK] Input creado\n";

    std::cout << "Input:\n";
    std::cout << input << "\n\n";


    // ============================================================
    // 3. Forward
    // ============================================================

    Tensor output = layer.forward(input);

    std::cout << "[OK] Forward ejecutado\n";


    // ============================================================
    // 4. Comprobar shape
    // ============================================================

    assert(output.shape().size() == 2);

    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 2);

    std::cout << "[OK] Output shape correcta: [2, 2]\n";


    // ============================================================
    // 5. Mostrar output
    // ============================================================

    std::cout << "\nOutput:\n";
    std::cout << output << "\n";


    // ============================================================
    // 6. Comprobar numel
    // ============================================================

    assert(output.numel() == 4);

    std::cout << "\n[OK] Output numel = "
              << output.numel()
              << "\n";


    // ============================================================
    // Resultado
    // ============================================================

    std::cout << "\n==========================================\n";
    std::cout << "       TODOS LOS TESTS PASARON            \n";
    std::cout << "==========================================\n";

    return 0;
}