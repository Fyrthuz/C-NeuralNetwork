#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <cmath>

#include "lib/tensor.h"
#include "lib/ops.h"


// ============================================================
// UTILIDADES PARA TESTS
// ============================================================

static bool almost_equal(
    float a,
    float b,
    float epsilon = 1e-5f
) {
    return std::fabs(a - b) <= epsilon;
}


// ============================================================
// MAIN
// ============================================================

int main() {

    std::cout << "==========================================\n";
    std::cout << "    SUITE DE PRUEBAS - TENSOR ENGINE     \n";
    std::cout << "==========================================\n";


    // ============================================================
    // TEST 1 - Constructor básico y Strides
    // ============================================================

    std::cout << "\n--- TEST 1: Constructor basico y Strides ---\n";

    Tensor t1({2, 3, 4}, DType::F32, Device::CPU);

    t1.print("t1 {2, 3, 4}");

    assert(t1.numel() == 24);

    assert(
        t1.strides()[0] == 12 &&
        t1.strides()[1] == 4 &&
        t1.strides()[2] == 1
    );

    assert(t1.data() != nullptr);

    std::cout << "-> PASADO: Dimensiones y strides correctos.\n";


    // ============================================================
    // TEST 2 - Deep Copy
    // ============================================================

    std::cout << "\n--- TEST 2: Constructor y Asignacion por Copia ---\n";

    Tensor t2 = t1;

    t2.print("t2 (copia de t1)");

    assert(t2.data() != t1.data());
    assert(t2.numel() == t1.numel());

    Tensor t3;

    t3 = t2;

    assert(t3.data() != t2.data());
    assert(t3.numel() == t2.numel());

    std::cout << "-> PASADO: Copias profundas con buffers independientes.\n";


    // ============================================================
    // TEST 3 - Move Semantics
    // ============================================================

    std::cout << "\n--- TEST 3: Semantica de Movimiento ---\n";

    void* ptr_original = t2.data();

    Tensor t_moved = std::move(t2);

    assert(t_moved.data() == ptr_original);
    assert(t2.data() == nullptr);
    assert(t2.numel() == 0);
    assert(t2.shape().empty());

    std::cout
        << "-> Move constructor: Buffer transferido y origen reseteado.\n";


    Tensor t_assign_moved;

    t_assign_moved = std::move(t3);

    assert(t3.data() == nullptr);
    assert(t3.numel() == 0);
    assert(t_assign_moved.numel() == 24);

    std::cout << "-> PASADO: Semantica de movimiento completada.\n";


    // ============================================================
    // TEST 4 - zeros()
    // ============================================================

    std::cout << "\n--- TEST 4: Tensor::zeros ---\n";

    Tensor z = Tensor::zeros({2, 3}, DType::I32);

    auto* z_ptr = static_cast<int32_t*>(z.data());

    for (uint64_t i = 0; i < z.numel(); ++i) {
        assert(z_ptr[i] == 0);
    }

    std::cout << "-> PASADO: Tensor inicializado con ceros.\n";


    // ============================================================
    // TEST 5 - ones()
    // ============================================================

    std::cout << "\n--- TEST 5: Tensor::ones ---\n";

    Tensor o = Tensor::ones({4}, DType::F32);

    auto* o_ptr = static_cast<float*>(o.data());

    for (uint64_t i = 0; i < o.numel(); ++i) {
        assert(almost_equal(o_ptr[i], 1.0f));
    }

    std::cout << "-> PASADO: Tensor inicializado con unos.\n";


    // ============================================================
    // TEST 6 - arange()
    // ============================================================

    std::cout << "\n--- TEST 6: Tensor::arange ---\n";

    // [0, 2, 4, 6, 8]

    Tensor a = Tensor::arange(0, 10, 2, DType::I64);

    auto* a_ptr = static_cast<int64_t*>(a.data());

    std::cout << "Tensor generado: " << a << std::endl;

    assert(a.numel() == 5);

    assert(
        a_ptr[0] == 0 &&
        a_ptr[2] == 4 &&
        a_ptr[4] == 8
    );

    std::cout << "-> PASADO: Rango secuencial generado con exito.\n";


    // ============================================================
    // TEST 7 - Excepciones
    // ============================================================

    std::cout << "\n--- TEST 7: Validacion de Excepciones ---\n";

    try {

        Tensor::arange(10, 0, 1);

        assert(false);

    } catch (const std::invalid_argument& e) {

        std::cout
            << "-> Excepcion esperada capturada: "
            << e.what()
            << "\n";
    }


    try {

        Tensor::zeros(
            {2, 2},
            DType::F32,
            Device::CUDA
        );

        assert(false);

    } catch (const std::runtime_error& e) {

        std::cout
            << "-> Excepcion esperada capturada: "
            << e.what()
            << "\n";
    }


    // ============================================================
    // TEST 8 - clone()
    // ============================================================

    std::cout << "\n--- TEST 8: Clone ---\n";

    Tensor b = a.clone();

    std::cout << "Original: " << a << std::endl;
    std::cout << "Clonado : " << b << std::endl;

    assert(b.data() != a.data());
    assert(b.numel() == a.numel());
    assert(b.shape() == a.shape());

    auto* b_ptr = static_cast<int64_t*>(b.data());

    for (uint64_t i = 0; i < b.numel(); ++i) {
        assert(b_ptr[i] == a_ptr[i]);
    }

    std::cout << "-> PASADO: Clone correcto.\n";


    // ============================================================
    // TEST 9 - flatten()
    // ============================================================

    std::cout << "\n--- TEST 9: Flatten ---\n";

    Tensor c = b.flatten();

    std::cout << "Flatten: " << c << std::endl;

    assert(c.numel() == b.numel());
    assert(c.shape().size() == 1);
    assert(c.shape()[0] == b.numel());

    std::cout << "-> PASADO: Flatten correcto.\n";


    // ============================================================
    // TEST 10 - transpose()
    // ============================================================

    std::cout << "\n--- TEST 10: Transpose ---\n";

    Tensor d = t1.transpose(1, 0);

    std::cout << "t1:\n";
    std::cout << t1 << std::endl;

    std::cout << "transpose(1, 0):\n";
    std::cout << d << std::endl;

    assert(d.shape().size() == 3);

    assert(d.shape()[0] == 3);
    assert(d.shape()[1] == 2);
    assert(d.shape()[2] == 4);

    std::cout << "-> PASADO: Transpose correcto.\n";


    // ============================================================
    // OPERACIONES ELEMENT-WISE
    // ============================================================


    // ============================================================
    // TEST 11 - ADD
    // ============================================================

    std::cout << "\n--- TEST 11: add() ---\n";

    Tensor x = Tensor::arange(
        1,
        5,
        1,
        DType::F32
    );

    Tensor y = Tensor::ones(
        {4},
        DType::F32
    );

    Tensor add_result = add(x, y);

    std::cout << "x          = " << x << std::endl;
    std::cout << "y          = " << y << std::endl;
    std::cout << "x + y      = " << add_result << std::endl;

    auto* add_ptr =
        static_cast<float*>(add_result.data());

    assert(almost_equal(add_ptr[0], 2.0f));
    assert(almost_equal(add_ptr[1], 3.0f));
    assert(almost_equal(add_ptr[2], 4.0f));
    assert(almost_equal(add_ptr[3], 5.0f));

    std::cout << "-> PASADO: add() correcto.\n";


    // ============================================================
    // TEST 12 - SUB
    // ============================================================

    std::cout << "\n--- TEST 12: sub() ---\n";

    Tensor sub_result = sub(x, y);

    std::cout << "x - y      = " << sub_result << std::endl;

    auto* sub_ptr =
        static_cast<float*>(sub_result.data());

    assert(almost_equal(sub_ptr[0], 0.0f));
    assert(almost_equal(sub_ptr[1], 1.0f));
    assert(almost_equal(sub_ptr[2], 2.0f));
    assert(almost_equal(sub_ptr[3], 3.0f));

    std::cout << "-> PASADO: sub() correcto.\n";


    // ============================================================
    // TEST 13 - MULT
    // ============================================================

    std::cout << "\n--- TEST 13: mult() ---\n";

    Tensor mult_result = mult(x, y);

    std::cout << "x * y      = " << mult_result << std::endl;

    auto* mult_ptr =
        static_cast<float*>(mult_result.data());

    assert(almost_equal(mult_ptr[0], 1.0f));
    assert(almost_equal(mult_ptr[1], 2.0f));
    assert(almost_equal(mult_ptr[2], 3.0f));
    assert(almost_equal(mult_ptr[3], 4.0f));

    std::cout << "-> PASADO: mult() correcto.\n";


    // ============================================================
    // TEST 14 - DIV
    // ============================================================

    std::cout << "\n--- TEST 14: div() ---\n";

    Tensor div_result = div(x, y);

    std::cout << "x / y      = " << div_result << std::endl;

    auto* div_ptr =
        static_cast<float*>(div_result.data());

    assert(almost_equal(div_ptr[0], 1.0f));
    assert(almost_equal(div_ptr[1], 2.0f));
    assert(almost_equal(div_ptr[2], 3.0f));
    assert(almost_equal(div_ptr[3], 4.0f));

    std::cout << "-> PASADO: div() correcto.\n";


    // ============================================================
    // MATMUL
    // ============================================================


    // ============================================================
    // TEST 15 - Matmul sencillo
    // ============================================================

    std::cout << "\n--- TEST 15: matmul() ---\n";

    /*
        A =

        [1 2 3]
        [4 5 6]

        Shape: [2, 3]


        B =

        [ 7  8 ]
        [ 9 10 ]
        [11 12]

        Shape: [3, 2]


        Resultado:

        [ 58  64 ]
        [139 154]

        Shape: [2, 2]
    */

    Tensor A({2, 3}, DType::F32, Device::CPU);
    Tensor B({3, 2}, DType::F32, Device::CPU);

    auto* A_ptr =
        static_cast<float*>(A.data());

    auto* B_ptr =
        static_cast<float*>(B.data());


    A_ptr[0] = 1.0f;
    A_ptr[1] = 2.0f;
    A_ptr[2] = 3.0f;

    A_ptr[3] = 4.0f;
    A_ptr[4] = 5.0f;
    A_ptr[5] = 6.0f;


    B_ptr[0] = 7.0f;
    B_ptr[1] = 8.0f;

    B_ptr[2] = 9.0f;
    B_ptr[3] = 10.0f;

    B_ptr[4] = 11.0f;
    B_ptr[5] = 12.0f;


    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;


    Tensor C = matmul(A, B);

    std::cout << "A @ B = " << C << std::endl;


    assert(C.shape().size() == 2);
    assert(C.shape()[0] == 2);
    assert(C.shape()[1] == 2);
    assert(C.numel() == 4);


    auto* C_ptr =
        static_cast<float*>(C.data());


    assert(almost_equal(C_ptr[0], 58.0f));
    assert(almost_equal(C_ptr[1], 64.0f));

    assert(almost_equal(C_ptr[2], 139.0f));
    assert(almost_equal(C_ptr[3], 154.0f));


    std::cout << "-> PASADO: matmul() correcto.\n";


    // ============================================================
    // TEST 16 - Matmul cuadrado
    // ============================================================

    std::cout << "\n--- TEST 16: Matmul cuadrado ---\n";

    Tensor M1({2, 2}, DType::F32, Device::CPU);
    Tensor M2({2, 2}, DType::F32, Device::CPU);


    auto* M1_ptr =
        static_cast<float*>(M1.data());

    auto* M2_ptr =
        static_cast<float*>(M2.data());


    // M1 =
    //
    // [1 2]
    // [3 4]

    M1_ptr[0] = 1.0f;
    M1_ptr[1] = 2.0f;
    M1_ptr[2] = 3.0f;
    M1_ptr[3] = 4.0f;


    // M2 =
    //
    // [5 6]
    // [7 8]

    M2_ptr[0] = 5.0f;
    M2_ptr[1] = 6.0f;
    M2_ptr[2] = 7.0f;
    M2_ptr[3] = 8.0f;


    Tensor M3 = matmul(M1, M2);

    std::cout << "M1 = " << M1 << std::endl;
    std::cout << "M2 = " << M2 << std::endl;
    std::cout << "M1 @ M2 = " << M3 << std::endl;


    auto* M3_ptr =
        static_cast<float*>(M3.data());

    /*
        Resultado:

        [19 22]
        [43 50]
    */

    assert(almost_equal(M3_ptr[0], 19.0f));
    assert(almost_equal(M3_ptr[1], 22.0f));
    assert(almost_equal(M3_ptr[2], 43.0f));
    assert(almost_equal(M3_ptr[3], 50.0f));


    std::cout << "-> PASADO: Matmul cuadrado correcto.\n";


    // ============================================================
    // TEST 17 - Matmul incompatible
    // ============================================================

    std::cout << "\n--- TEST 17: Matmul incompatible ---\n";

    Tensor invalid_A({2, 3}, DType::F32, Device::CPU);
    Tensor invalid_B({4, 2}, DType::F32, Device::CPU);

    try {

        Tensor invalid_result =
            matmul(invalid_A, invalid_B);

        assert(false);

    } catch (const std::invalid_argument& e) {

        std::cout
            << "-> Excepcion esperada capturada: "
            << e.what()
            << "\n";
    }


    // ============================================================
    // TEST 18 - Operaciones con shapes incompatibles
    // ============================================================

    std::cout
        << "\n--- TEST 18: Shapes incompatibles ---\n";

    Tensor wrong_shape({2, 2}, DType::F32, Device::CPU);

    try {

        Tensor invalid = add(x, wrong_shape);

        assert(false);

    } catch (const std::invalid_argument& e) {

        std::cout
            << "-> add(): excepcion esperada: "
            << e.what()
            << "\n";
    }


    // ============================================================
    // TEST 19 - Dtypes incompatibles
    // ============================================================

    std::cout
        << "\n--- TEST 19: Dtypes incompatibles ---\n";

    Tensor float_tensor(
        {4},
        DType::F32,
        Device::CPU
    );

    Tensor int_tensor(
        {4},
        DType::I32,
        Device::CPU
    );

    try {

        Tensor invalid =
            add(float_tensor, int_tensor);

        assert(false);

    } catch (const std::invalid_argument& e) {

        std::cout
            << "-> add(): excepcion esperada: "
            << e.what()
            << "\n";
    }


    // ============================================================
    // BROADCASTING
    // ============================================================


    // ============================================================
    // TEST 20 - Broadcasting [2,3] + [3]
    // ============================================================

    std::cout
        << "\n--- TEST 20: Broadcasting [2,3] + [3] ---\n";

    /*
        x =
        [1 2 3]
        [4 5 6]

        y =
        [10 20 30]

        Resultado:

        [11 22 33]
        [14 25 36]
    */

    Tensor bx({2, 3}, DType::F32, Device::CPU);
    Tensor by({3}, DType::F32, Device::CPU);

    auto* bx_ptr =
        bx.data_ptr<float>();

    auto* by_ptr =
        by.data_ptr<float>();


    bx_ptr[0] = 1.0f;
    bx_ptr[1] = 2.0f;
    bx_ptr[2] = 3.0f;

    bx_ptr[3] = 4.0f;
    bx_ptr[4] = 5.0f;
    bx_ptr[5] = 6.0f;


    by_ptr[0] = 10.0f;
    by_ptr[1] = 20.0f;
    by_ptr[2] = 30.0f;


    Tensor broadcast_add =
        add(bx, by);

    std::cout
        << "bx              = "
        << bx
        << std::endl;

    std::cout
        << "by              = "
        << by
        << std::endl;

    std::cout
        << "bx + by         = "
        << broadcast_add
        << std::endl;


    assert(
        broadcast_add.shape() ==
        std::vector<int64_t>({2, 3})
    );


    auto* broadcast_add_ptr =
        broadcast_add.data_ptr<float>();


    assert(almost_equal(
        broadcast_add_ptr[0], 11.0f
    ));

    assert(almost_equal(
        broadcast_add_ptr[1], 22.0f
    ));

    assert(almost_equal(
        broadcast_add_ptr[2], 33.0f
    ));

    assert(almost_equal(
        broadcast_add_ptr[3], 14.0f
    ));

    assert(almost_equal(
        broadcast_add_ptr[4], 25.0f
    ));

    assert(almost_equal(
        broadcast_add_ptr[5], 36.0f
    ));


    std::cout
        << "-> PASADO: Broadcasting [2,3] + [3].\n";


    // ============================================================
    // TEST 21 - Broadcasting [2,3] + [2,1]
    // ============================================================

    std::cout
        << "\n--- TEST 21: Broadcasting [2,3] + [2,1] ---\n";

    /*
        x =
        [1 2 3]
        [4 5 6]

        y =
        [10]
        [20]

        Resultado:

        [11 12 13]
        [24 25 26]
    */

    Tensor bx2({2, 3}, DType::F32, Device::CPU);
    Tensor by2({2, 1}, DType::F32, Device::CPU);

    auto* bx2_ptr =
        bx2.data_ptr<float>();

    auto* by2_ptr =
        by2.data_ptr<float>();


    bx2_ptr[0] = 1.0f;
    bx2_ptr[1] = 2.0f;
    bx2_ptr[2] = 3.0f;

    bx2_ptr[3] = 4.0f;
    bx2_ptr[4] = 5.0f;
    bx2_ptr[5] = 6.0f;


    by2_ptr[0] = 10.0f;
    by2_ptr[1] = 20.0f;


    Tensor broadcast_add2 =
        add(bx2, by2);

    std::cout
        << "bx2             = "
        << bx2
        << std::endl;

    std::cout
        << "by2             = "
        << by2
        << std::endl;

    std::cout
        << "bx2 + by2       = "
        << broadcast_add2
        << std::endl;


    assert(
        broadcast_add2.shape() ==
        std::vector<int64_t>({2, 3})
    );


    auto* broadcast_add2_ptr =
        broadcast_add2.data_ptr<float>();


    assert(almost_equal(
        broadcast_add2_ptr[0], 11.0f
    ));

    assert(almost_equal(
        broadcast_add2_ptr[1], 12.0f
    ));

    assert(almost_equal(
        broadcast_add2_ptr[2], 13.0f
    ));

    assert(almost_equal(
        broadcast_add2_ptr[3], 24.0f
    ));

    assert(almost_equal(
        broadcast_add2_ptr[4], 25.0f
    ));

    assert(almost_equal(
        broadcast_add2_ptr[5], 26.0f
    ));


    std::cout
        << "-> PASADO: Broadcasting [2,3] + [2,1].\n";


    // ============================================================
    // TEST 22 - Broadcasting [2,3] - [3]
    // ============================================================

    std::cout
        << "\n--- TEST 22: Broadcasting [2,3] - [3] ---\n";

    /*
        x =
        [1 2 3]
        [4 5 6]

        y =
        [10 20 30]

        Resultado:

        [-9 -18 -27]
        [-6 -15 -24]
    */

    Tensor sub_broadcast =
        sub(bx, by);

    std::cout
        << "bx - by         = "
        << sub_broadcast
        << std::endl;


    assert(
        sub_broadcast.shape() ==
        std::vector<int64_t>({2, 3})
    );


    const float* sub_broadcast_ptr =
        sub_broadcast.data_ptr<float>();


    assert(almost_equal(
        sub_broadcast_ptr[0], -9.0f
    ));

    assert(almost_equal(
        sub_broadcast_ptr[1], -18.0f
    ));

    assert(almost_equal(
        sub_broadcast_ptr[2], -27.0f
    ));

    assert(almost_equal(
        sub_broadcast_ptr[3], -6.0f
    ));

    assert(almost_equal(
        sub_broadcast_ptr[4], -15.0f
    ));

    assert(almost_equal(
        sub_broadcast_ptr[5], -24.0f
    ));


    std::cout
        << "-> PASADO: Broadcasting [2,3] - [3].\n";


    // ============================================================
    // TEST 23 - Broadcasting [2,3] - [2,1]
    // ============================================================

    std::cout
        << "\n--- TEST 23: Broadcasting [2,3] - [2,1] ---\n";

    /*
        x =
        [1 2 3]
        [4 5 6]

        y =
        [10]
        [20]

        Resultado:

        [-9 -8 -7]
        [-16 -15 -14]
    */

    Tensor sub_broadcast2 =
        sub(bx2, by2);

    std::cout
        << "bx2 - by2       = "
        << sub_broadcast2
        << std::endl;


    assert(
        sub_broadcast2.shape() ==
        std::vector<int64_t>({2, 3})
    );


    const float* sub_broadcast2_ptr =
        sub_broadcast2.data_ptr<float>();


    assert(almost_equal(
        sub_broadcast2_ptr[0], -9.0f
    ));

    assert(almost_equal(
        sub_broadcast2_ptr[1], -8.0f
    ));

    assert(almost_equal(
        sub_broadcast2_ptr[2], -7.0f
    ));

    assert(almost_equal(
        sub_broadcast2_ptr[3], -16.0f
    ));

    assert(almost_equal(
        sub_broadcast2_ptr[4], -15.0f
    ));

    assert(almost_equal(
        sub_broadcast2_ptr[5], -14.0f
    ));


    std::cout
        << "-> PASADO: Broadcasting [2,3] - [2,1].\n";


    // ============================================================
    // TEST 24 - Broadcasting [2,3] * [3]
    // ============================================================

    std::cout
        << "\n--- TEST 24: Broadcasting [2,3] * [3] ---\n";

    /*
        x =
        [1 2 3]
        [4 5 6]

        y =
        [10 20 30]

        Resultado:

        [10 40 90]
        [40 100 180]
    */

    Tensor mult_broadcast =
        mult(bx, by);

    std::cout
        << "bx * by         = "
        << mult_broadcast
        << std::endl;


    assert(
        mult_broadcast.shape() ==
        std::vector<int64_t>({2, 3})
    );


    const float* mult_broadcast_ptr =
        mult_broadcast.data_ptr<float>();


    assert(almost_equal(
        mult_broadcast_ptr[0], 10.0f
    ));

    assert(almost_equal(
        mult_broadcast_ptr[1], 40.0f
    ));

    assert(almost_equal(
        mult_broadcast_ptr[2], 90.0f
    ));

    assert(almost_equal(
        mult_broadcast_ptr[3], 40.0f
    ));

    assert(almost_equal(
        mult_broadcast_ptr[4], 100.0f
    ));

    assert(almost_equal(
        mult_broadcast_ptr[5], 180.0f
    ));


    std::cout
        << "-> PASADO: Broadcasting [2,3] * [3].\n";


    // ============================================================
    // TEST 25 - Broadcasting [2,3] / [2,1]
    // ============================================================

    std::cout
        << "\n--- TEST 25: Broadcasting [2,3] / [2,1] ---\n";

    /*
        x =
        [1 2 3]
        [4 5 6]

        y =
        [10]
        [20]

        Resultado:

        [0.1  0.2  0.3]
        [0.2  0.25 0.3]
    */

    Tensor div_broadcast =
        div(bx2, by2);

    std::cout
        << "bx2 / by2       = "
        << div_broadcast
        << std::endl;


    assert(
        div_broadcast.shape() ==
        std::vector<int64_t>({2, 3})
    );


    const float* div_broadcast_ptr =
        div_broadcast.data_ptr<float>();


    assert(almost_equal(
        div_broadcast_ptr[0], 0.1f
    ));

    assert(almost_equal(
        div_broadcast_ptr[1], 0.2f
    ));

    assert(almost_equal(
        div_broadcast_ptr[2], 0.3f
    ));

    assert(almost_equal(
        div_broadcast_ptr[3], 0.2f
    ));

    assert(almost_equal(
        div_broadcast_ptr[4], 0.25f
    ));

    assert(almost_equal(
        div_broadcast_ptr[5], 0.3f
    ));


    std::cout
        << "-> PASADO: Broadcasting [2,3] / [2,1].\n";


    // ============================================================
    // TEST 26 - Broadcasting incompatible
    // ============================================================

    std::cout
        << "\n--- TEST 26: Broadcasting incompatible ---\n";

    Tensor incompatible_a(
        {2, 3},
        DType::F32,
        Device::CPU
    );

    Tensor incompatible_b(
        {2, 2},
        DType::F32,
        Device::CPU
    );

    try {

        Tensor invalid =
            add(incompatible_a, incompatible_b);

        // Si llegamos aquí, el broadcasting
        // ha aceptado incorrectamente las shapes.

        assert(false);

    } catch (const std::invalid_argument& e) {

        std::cout
            << "-> Excepcion esperada capturada: "
            << e.what()
            << "\n";
    }


    // ============================================================
    // FIN
    // ============================================================

    std::cout
        << "\n==========================================\n";

    std::cout
        << "   TODAS LAS PRUEBAS FINALIZARON CON EXITO!\n";

    std::cout
        << "==========================================\n";


    return 0;
}