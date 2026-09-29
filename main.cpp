#include <iostream>
#include <vector> 
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstring>
#include "lib/tensor.h"



int main() {
    std::cout << "==========================================\n";
    std::cout << "    SUITE DE PRUEBAS - TENSOR ENGINE     \n";
    std::cout << "==========================================\n";

    // 1. Constructor básico y cálculo de Strides (Row-Major)
    std::cout << "\n--- TEST 1: Constructor basico y Strides ---";
    Tensor t1({2, 3, 4}, DType::F32, Device::CPU);
    t1.print("t1 {2, 3, 4}");
    assert(t1.numel() == 24);
    assert(t1.strides()[0] == 12 && t1.strides()[1] == 4 && t1.strides()[2] == 1);
    assert(t1.data() != nullptr);
    std::cout << "-> PASADO: Dimensiones y strides correctos.\n";

    // 2. Semántica de Copia (Deep Copy)
    std::cout << "\n--- TEST 2: Constructor y Asignacion por Copia ---";
    Tensor t2 = t1; // Constructor de copia
    t2.print("t2 (copia de t1)");
    assert(t2.data() != t1.data()); // Punteros distintos (memoria independiente)
    assert(t2.numel() == t1.numel());

    Tensor t3;
    t3 = t2; // Operador de asignación por copia
    assert(t3.data() != t2.data());
    assert(t3.numel() == t2.numel());
    std::cout << "-> PASADO: Copias profundas con buffers independientes.\n";

    // 3. Semántica de Movimiento (Move Semantics)
    std::cout << "\n--- TEST 3: Semantica de Movimiento ---";
    void* ptr_original = t2.data();
    Tensor t_moved = std::move(t2); // Constructor de movimiento
    assert(t_moved.data() == ptr_original);
    assert(t2.data() == nullptr);
    assert(t2.numel() == 0);
    assert(t2.shape().empty());
    std::cout << "-> Move constructor: Buffer transferido y origen reseteado a nullptr.\n";

    Tensor t_assign_moved;
    t_assign_moved = std::move(t3); // Asignación por movimiento
    assert(t3.data() == nullptr);
    assert(t3.numel() == 0);
    assert(t_assign_moved.numel() == 24);
    std::cout << "-> PASADO: Semantica de movimiento completada.\n";

    // 4. Factoría: zeros()
    std::cout << "\n--- TEST 4: Tensor::zeros ---";
    Tensor z = Tensor::zeros({2, 3}, DType::I32);
    auto* z_ptr = static_cast<int32_t *>(z.data());
    for (uint64_t i = 0; i < z.numel(); ++i) {
        assert(z_ptr[i] == 0);
    }
    std::cout << "-> PASADO: Tensor inicializado con ceros.\n";

    // 5. Factoría: ones()
    std::cout << "\n--- TEST 5: Tensor::ones ---";
    Tensor o = Tensor::ones({4}, DType::F32);
    auto* o_ptr = static_cast<float *>(o.data());
    for (uint64_t i = 0; i < o.numel(); ++i) {
        assert(o_ptr[i] == 1.0f);
    }
    std::cout << "-> PASADO: Tensor inicializado con unos.\n";

    // 6. Factoría: arange()
    std::cout << "\n--- TEST 6: Tensor::arange ---";
    // Genera: [0, 2, 4, 6, 8]
    Tensor a = Tensor::arange(0, 10, 2, DType::I64);
    auto* a_ptr = static_cast<int64_t *>(a.data());
    std::cout << "Tensor generado" << a << std::endl;
    assert(a.numel() == 5);
    assert(a_ptr[0] == 0 && a_ptr[2] == 4 && a_ptr[4] == 8);
    std::cout << "-> PASADO: Rango secuencial generado con exito.\n";

    // 7. Gestión de errores y excepciones
    std::cout << "\n--- TEST 7: Validacion de Excepciones ---";
    try {
        Tensor::arange(10, 0, 1); // Rango inválido
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "-> Excepcion esperada capturada: " << e.what() << "\n";
    }

    try {
        Tensor::zeros({2, 2}, DType::F32, Device::CUDA); // Dispositivo no disponible
        assert(false);
    } catch (const std::runtime_error& e) {
        std::cout << "-> Excepcion esperada capturada: " << e.what() << "\n";
    }


    // 6. Factoría: arange()
    std::cout << "\n--- TEST 6: Tensor::arange ---";
    // Genera: [0, 2, 4, 6, 8]
    std::cout << "Tensor generado" << a << std::endl;
    std::cout << "Probando clonacion" << std::endl;
    Tensor b = a.clone();
    std::cout << "Vector clonado probando flatten..." << b << std::endl;
    Tensor c = b.flatten();
    std::cout << "Vector flatteneado" << c << std::endl;
    Tensor d = t1.transpose(1, 0);
    std::cout << "Vector transposeado" << d << std::endl;
    std::cout << "-> PASADO: Clonacion y flatten exitosa.\n";


    std::cout << "\n==========================================\n";
    std::cout << "   ¡TODAS LAS PRUEBAS FINALIZARON CON EXITO!\n";
    std::cout << "==========================================\n";


    Tensor t6 = Tensor({2, 2}, DType::F32, Device::CPU);
    std::cout << t6 << std::endl;
    return 0;
}