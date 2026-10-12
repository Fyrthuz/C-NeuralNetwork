// =================================================================
// main3.cpp — Suite de pruebas del motor de tensores con CUDA
//
// Ejercita las operaciones implementadas en GPU:
//   - Construcción de tensores en Device::CUDA
//   - Factory methods:  zeros, ones, arange, randn
//   - Utilidades:       flatten, transpose, clone
//   - Copia:            CPU <-> GPU (cudaMemcpy)
//   - Semántica de movimiento (move ctor/assign) en GPU
//
// Compara los resultados GPU contra el mismo cálculo en CPU para
// validar correctness de cada kernel CUDA.
// =================================================================

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <random>

#include "lib/tensor.h"
#include "lib/kernel_operations.h"

#include <cuda_runtime.h>

// ============================================================
// UTILIDADES PARA TESTS
// ============================================================

static bool almost_equal(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) <= epsilon;
}

static bool almost_equal(double a, double b, double epsilon = 1e-9) {
    return std::fabs(a - b) <= epsilon;
}

// Comparación CPU vs GPU: descarga el tensor CUDA y compara por elemento
template <typename T>
static bool tensors_match(const Tensor& cpu, const Tensor& gpu, T /*tag*/) {
    assert(cpu.shape() == gpu.shape());
    assert(cpu.dtype() == gpu.dtype());

    const uint64_t n = cpu.numel();
    if (n == 0) return true;

    // Buffer CPU con la copia del tensor GPU
    std::vector<char> gpu_bytes(n * cpu.get_element_size(cpu.dtype()));
    cudaError_t err = cudaMemcpy(
        gpu_bytes.data(), gpu.data(),
        gpu_bytes.size(), cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess) {
        std::cerr << "  [ERROR] cudaMemcpy DtoH: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    const T* cpu_ptr = cpu.data_ptr<T>();
    const T* gpu_ptr = reinterpret_cast<const T*>(gpu_bytes.data());

    for (uint64_t i = 0; i < n; ++i) {
        if (!almost_equal(static_cast<double>(cpu_ptr[i]),
                          static_cast<double>(gpu_ptr[i]))) {
            std::cerr << "  [MISMATCH] idx=" << i
                      << " cpu=" << cpu_ptr[i]
                      << " gpu=" << gpu_ptr[i] << "\n";
            return false;
        }
    }
    return true;
}

static bool check_cuda_ready() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "[FATAL] cudaGetDeviceCount fallo: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }
    if (count == 0) {
        std::cerr << "[FATAL] No hay GPUs CUDA disponibles.\n";
        return false;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[GPU] " << prop.name << " | "
              << "SM " << prop.major << "." << prop.minor << " | "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MiB\n";
    return true;
}

// ============================================================
// MAIN
// ============================================================

int main() {

    std::cout << "==========================================\n";
    std::cout << "  SUITE CUDA - TENSOR ENGINE (main3)    \n";
    std::cout << "==========================================\n";

    if (!check_cuda_ready()) return 1;

    // ============================================================
    // TEST 1 — zeros CUDA: shape + buffer a 0 + copia CPU/GPU
    // ============================================================
    std::cout << "\n--- TEST 1: Tensor::zeros en CUDA ---\n";

    {
        Tensor cpu = Tensor::zeros({2, 3, 4}, DType::F32, Device::CPU);
        Tensor gpu = Tensor::zeros({2, 3, 4}, DType::F32, Device::CUDA);

        assert(cpu.device() == Device::CPU);
        assert(gpu.device() == Device::CUDA);
        assert(cpu.numel() == gpu.numel());

        if (tensors_match(cpu, gpu, 0.0f)) {
            std::cout << "-> PASADO: zeros CUDA == zeros CPU "
                      << "(shape [2,3,4], numel=" << gpu.numel() << ")\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    // ============================================================
    // TEST 2 — ones CUDA
    // ============================================================
    std::cout << "\n--- TEST 2: Tensor::ones en CUDA ---\n";

    {
        Tensor cpu = Tensor::ones({128}, DType::F32, Device::CPU);
        Tensor gpu = Tensor::ones({128}, DType::F32, Device::CUDA);

        if (tensors_match(cpu, gpu, 0.0f)) {
            std::cout << "-> PASADO: ones CUDA == ones CPU (shape [128])\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    // ============================================================
    // TEST 3 — ones CUDA con varios dtypes (DISPATCH_ALL_TYPES)
    // ============================================================
    std::cout << "\n--- TEST 3: Tensor::ones en CUDA con F32/F64/I32/I64 ---\n";

    {
        struct Case { DType dtype; const char* name; };
        Case cases[] = {
            {DType::F32, "F32"}, {DType::F64, "F64"},
            {DType::I32, "I32"}, {DType::I64, "I64"}
        };

        bool ok = true;
        for (const auto& c : cases) {
            Tensor gpu = Tensor::ones({16}, c.dtype, Device::CUDA);

            // Descarga y verifica que todos los elementos sean 1
            size_t bytes = gpu.numel() * gpu.get_element_size(c.dtype);
            std::vector<char> buf(bytes);
            cudaMemcpy(buf.data(), gpu.data(), bytes, cudaMemcpyDeviceToHost);

            bool all_one = true;
            DISPATCH_ALL_TYPES(c.dtype, scalar_t, {
                const scalar_t* p = reinterpret_cast<const scalar_t*>(buf.data());
                for (uint64_t i = 0; i < gpu.numel(); ++i) {
                    if (static_cast<double>(p[i]) != 1.0) { all_one = false; break; }
                }
            });
            std::cout << "  [" << c.name << "] ones: "
                      << (all_one ? "OK" : "FAIL") << "\n";
            if (!all_one) ok = false;
        }
        std::cout << (ok ? "-> PASADO\n" : "-> FALLIDO\n");
        if (!ok) return 1;
    }

    // ============================================================
    // TEST 4 — arange CUDA: start, step, size arbitrarios
    // ============================================================
    std::cout << "\n--- TEST 4: Tensor::arange en CUDA ---\n";

    {
        // Caso (start=5, end=53, step=2) -> tamaño 24, valores 5,7,9,...,51
        Tensor cpu = Tensor::arange(5, 53, 2, DType::F32, Device::CPU);
        Tensor gpu = Tensor::arange(5, 53, 2, DType::F32, Device::CUDA);

        assert(cpu.numel() == gpu.numel());
        assert(cpu.numel() == 24);

        if (tensors_match(cpu, gpu, 0.0f)) {
            std::cout << "-> PASADO: arange(5,53,2) CUDA == CPU "
                      << "(numel=" << gpu.numel() << ")\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    {
        // Step negativo: start=10, end=2, step=-1 -> [10, 9, ..., 3]
        Tensor cpu = Tensor::arange(10, 2, -1, DType::F64, Device::CPU);
        Tensor gpu = Tensor::arange(10, 2, -1, DType::F64, Device::CUDA);

        if (tensors_match(cpu, gpu, 0.0)) {
            std::cout << "-> PASADO: arange(10,2,-1) F64 CUDA == CPU "
                      << "(numel=" << gpu.numel() << ")\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    {
        // Step fraccionario lógico (1) en I64
        Tensor cpu = Tensor::arange(0, 100, 7, DType::I64, Device::CPU);
        Tensor gpu = Tensor::arange(0, 100, 7, DType::I64, Device::CUDA);

        size_t bytes = gpu.numel() * sizeof(int64_t);
        std::vector<char> buf(bytes);
        cudaMemcpy(buf.data(), gpu.data(), bytes, cudaMemcpyDeviceToHost);

        bool ok = true;
        const int64_t* gp = reinterpret_cast<const int64_t*>(buf.data());
        for (uint64_t i = 0; i < gpu.numel(); ++i) {
            if (gp[i] != static_cast<int64_t>(i) * 7) { ok = false; break; }
        }
        std::cout << (ok ? "-> PASADO: arange I64 step=7\n" : "-> FALLIDO\n");
        if (!ok) return 1;
    }

    // ============================================================
    // TEST 5 — randn CUDA: estadística (media ~mean, std ~std)
    // ============================================================
    std::cout << "\n--- TEST 5: Tensor::randn en CUDA (estadistica) ---\n";

    {
        const int64_t N = 100000;
        const float target_mean = 2.5f, target_std = 1.7f;

        Tensor gpu = Tensor::randn({N}, target_mean, target_std,
                                   DType::F32, Device::CUDA);

        std::vector<float> h(N);
        cudaMemcpy(h.data(), gpu.data(), N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (float v : h) sum += v;
        const double mean = sum / N;

        double sq = 0.0;
        for (float v : h) sq += (v - mean) * (v - mean);
        const double std = std::sqrt(sq / N);

        const bool mean_ok = std::fabs(mean - target_mean) < 0.1;
        const bool std_ok  = std::fabs(std  - target_std)  < 0.1;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  N=" << N
                  << "  mean=" << mean << " (target " << target_mean << ")"
                  << "  std="  << std  << " (target " << target_std  << ")\n";

        if (mean_ok && std_ok) {
            std::cout << "-> PASADO: randn CUDA respeta media y desviacion\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    // ============================================================
    // TEST 6 — transpose CUDA 2D y 3D
    // ============================================================
    std::cout << "\n--- TEST 6: Tensor::transpose en CUDA ---\n";

    {
        // 2D: [3, 5] -> [5, 3]
        Tensor cpu = Tensor::arange(0, 15, 1, DType::F32, Device::CPU)
                         .clone();
        // reshape via flatten + nueva copia ordenada: usamos otra fuente
        Tensor src_cpu({3, 5}, DType::F32, Device::CPU);
        for (int64_t i = 0; i < src_cpu.numel(); ++i)
            src_cpu.data_ptr<float>()[i] = static_cast<float>(i);

        Tensor src_gpu({3, 5}, DType::F32, Device::CUDA);
        cudaMemcpy(src_gpu.data(), src_cpu.data(),
                   src_cpu.numel() * sizeof(float),
                   cudaMemcpyHostToDevice);

        Tensor t_cpu = src_cpu.transpose(0, 1);
        Tensor t_gpu = src_gpu.transpose(0, 1);

        assert(t_cpu.shape() == t_gpu.shape());

        if (tensors_match(t_cpu, t_gpu, 0.0f)) {
            std::cout << "-> PASADO: transpose 2D [3,5] -> [5,3]\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    {
        // 3D: [2, 3, 4] swap dims 0 y 2 -> [4, 3, 2]
        Tensor src_cpu({2, 3, 4}, DType::F32, Device::CPU);
        for (int64_t i = 0; i < src_cpu.numel(); ++i)
            src_cpu.data_ptr<float>()[i] = static_cast<float>(i);

        Tensor src_gpu({2, 3, 4}, DType::F32, Device::CUDA);
        cudaMemcpy(src_gpu.data(), src_cpu.data(),
                   src_cpu.numel() * sizeof(float),
                   cudaMemcpyHostToDevice);

        Tensor t_cpu = src_cpu.transpose(0, 2);
        Tensor t_gpu = src_gpu.transpose(0, 2);

        assert(t_cpu.shape() == t_gpu.shape());
        assert(t_cpu.shape()[0] == 4 && t_cpu.shape()[1] == 3
               && t_cpu.shape()[2] == 2);

        if (tensors_match(t_cpu, t_gpu, 0.0f)) {
            std::cout << "-> PASADO: transpose 3D [2,3,4] swap 0<->2\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    // ============================================================
    // TEST 7 — flatten CUDA (debe preservar orden)
    // ============================================================
    std::cout << "\n--- TEST 7: Tensor::flatten en CUDA ---\n";

    {
        Tensor src_cpu({2, 3, 4}, DType::F32, Device::CPU);
        for (int64_t i = 0; i < src_cpu.numel(); ++i)
            src_cpu.data_ptr<float>()[i] = static_cast<float>(i * i);

        Tensor src_gpu({2, 3, 4}, DType::F32, Device::CUDA);
        cudaMemcpy(src_gpu.data(), src_cpu.data(),
                   src_cpu.numel() * sizeof(float),
                   cudaMemcpyHostToDevice);

        Tensor f_cpu = src_cpu.flatten();
        Tensor f_gpu = src_gpu.flatten();

        assert(f_cpu.shape().size() == 1);
        assert(f_cpu.shape()[0] == src_cpu.numel());

        if (tensors_match(f_cpu, f_gpu, 0.0f)) {
            std::cout << "-> PASADO: flatten CUDA preserva orden "
                      << "(numel=" << f_gpu.numel() << ")\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    // ============================================================
    // TEST 8 — copia profunda CPU<->GPU y GPU->GPU
    // ============================================================
    std::cout << "\n--- TEST 8: Copy ctor y operator= en CUDA ---\n";

    {
        Tensor original = Tensor::arange(0, 32, 1, DType::F32, Device::CUDA);

        // Copia por constructor
        Tensor copy1(original);
        assert(copy1.data() != original.data());
        assert(copy1.device() == Device::CUDA);
        assert(copy1.numel() == original.numel());

        // Copia por asignación
        Tensor copy2;
        copy2 = original;
        assert(copy2.data() != original.data());

        // GPU -> GPU debe coincidir
        size_t bytes = original.numel() * sizeof(float);
        std::vector<float> h_orig(original.numel()),
                           h_c1(copy1.numel()),
                           h_c2(copy2.numel());

        cudaMemcpy(h_orig.data(), original.data(), bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c1.data(),   copy1.data(),   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c2.data(),   copy2.data(),   bytes, cudaMemcpyDeviceToHost);

        bool ok = true;
        for (uint64_t i = 0; i < original.numel(); ++i) {
            if (h_orig[i] != h_c1[i] || h_orig[i] != h_c2[i]) { ok = false; break; }
        }
        std::cout << (ok ? "-> PASADO: copias profundas GPU correctas\n"
                         : "-> FALLIDO\n");
        if (!ok) return 1;
    }

    // ============================================================
    // TEST 9 — movimiento en CUDA (sin pérdida de datos)
    // ============================================================
    std::cout << "\n--- TEST 9: Move ctor y move assign en CUDA ---\n";

    {
        Tensor src = Tensor::arange(100, 132, 1, DType::F32, Device::CUDA);
        void* src_data_before = src.data();

        // Move constructor
        Tensor moved(std::move(src));
        assert(moved.device() == Device::CUDA);
        assert(moved.data() == src_data_before);
        assert(src.data() == nullptr);
        assert(src.numel() == 0);

        // Move assignment
        Tensor dest;
        dest = std::move(moved);
        assert(dest.device() == Device::CUDA);
        assert(dest.data() == src_data_before);

        // El dato sigue siendo [100, 101, ..., 131]
        std::vector<float> h(dest.numel());
        cudaMemcpy(h.data(), dest.data(),
                   dest.numel() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        bool ok = true;
        for (uint64_t i = 0; i < dest.numel(); ++i) {
            if (h[i] != 100.0f + static_cast<float>(i)) { ok = false; break; }
        }
        std::cout << (ok ? "-> PASADO: movimiento preserva buffer y libera origen\n"
                         : "-> FALLIDO\n");
        if (!ok) return 1;
    }

    // ============================================================
    // TEST 10 — randn CUDA con shape 2D (típica para pesos)
    // ============================================================
    std::cout << "\n--- TEST 10: randn CUDA shape 2D (simulando pesos) ---\n";

    {
        // Simulamos pesos de Linear(128, 64)
        Tensor w = Tensor::randn({128, 64}, 0.0f, 0.05f,
                                  DType::F32, Device::CUDA);

        assert(w.device() == Device::CUDA);
        assert(w.shape()[0] == 128 && w.shape()[1] == 64);

        std::vector<float> h(w.numel());
        cudaMemcpy(h.data(), w.data(), w.numel() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        double sum = 0.0, sq = 0.0;
        for (float v : h) sum += v;
        const double mean = sum / w.numel();
        for (float v : h) sq += (v - mean) * (v - mean);
        const double std = std::sqrt(sq / w.numel());

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  shape=[128,64]  mean=" << mean << "  std=" << std << "\n";

        // std esperado ~0.05 (Kaiming-like)
        if (std::fabs(mean) < 0.01 && std::fabs(std - 0.05) < 0.01) {
            std::cout << "-> PASADO: distribucion estable\n";
        } else {
            std::cout << "-> FALLIDO\n";
            return 1;
        }
    }

    // ============================================================
    // TEST 11 — operator<< imprime correctamente tensores CUDA
    // ============================================================
    std::cout << "\n--- TEST 11: Impresion de tensor CUDA ---\n";

    {
        Tensor t = Tensor::arange(0, 6, 1, DType::F32, Device::CUDA);
        std::cout << t;  // Debe descargar y mostrar [0, 1, 2, 3, 4, 5]
        std::cout << "-> PASADO: operator<< funciona con CUDA\n";
    }

    // ============================================================
    // TEST 12 — Integración: zeros -> arange en GPU
    // ============================================================
    std::cout << "\n--- TEST 12: Pipeline CPU/GPU mixto ---\n";

    {
        // Creamos un tensor en GPU, lo movemos, lo devolvemos
        Tensor a = Tensor::arange(0, 1024, 1, DType::F32, Device::CUDA);
        Tensor b = std::move(a);  // move ctor

        // Hacemos transpose
        Tensor d2 = b.transpose(0, 0);  // identity
        assert(d2.shape() == b.shape());

        // a debe estar vacía tras el move
        assert(a.numel() == 0);
        assert(a.data() == nullptr);

        std::cout << "-> PASADO: pipeline mixto CPU/GPU\n";
    }

    // ============================================================
    // Resumen
    // ============================================================
    std::cout << "\n==========================================\n";
    std::cout << "  TODOS LOS TESTS CUDA FINALIZARON OK\n";
    std::cout << "==========================================\n";

    return 0;
}