
#include "kernel_operations.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <string>
#include <random>

constexpr int MAX_DIMS = 5;

template <typename scalar_t>
__global__ void arange_kernel(scalar_t* out, int64_t start, int64_t step, int64_t size){
    // Indice del bloque * Tamaño de cada bloque + Indice del hilo
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<scalar_t>(start + idx * step);
    }

}


void cuda_arange(void* out_ptr, int64_t start, int64_t step, int64_t size, DType dtype){

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        auto *ptr = static_cast<scalar_t *>(out_ptr);
        int threads = 256;
        int blocks = static_cast<int>((size + threads - 1) / threads);

        arange_kernel<scalar_t><<<blocks, threads>>>(ptr, start, step, size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Error en cuda_arange: ") + cudaGetErrorString(err));
        }
    });
}



template <typename scalar_t>
__global__ void fill_kernel(scalar_t* out_ptr, scalar_t value, int64_t size){
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out_ptr[idx] = value;
    }
}



void cuda_fill(void* out_ptr, double value, int64_t size, DType dtype){
    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        auto *ptr = static_cast<scalar_t *>(out_ptr);
        int threads = 256;
        int blocks = static_cast<int>((size + threads - 1) / threads);

        fill_kernel<scalar_t><<<blocks, threads>>>(ptr, static_cast<scalar_t>(value), size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Error en cuda_fill: ") + cudaGetErrorString(err));
        }
    });
}

template<typename scalar_t>
__global__ void transpose_kernel(scalar_t* __restrict__ out_ptr, const scalar_t* __restrict__ src_ptr, uint64_t numel, int ndim, const int64_t* __restrict__ new_shape, const int64_t* __restrict__ new_strides){
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    int64_t temp = idx;
    int64_t src_offset = 0;

    #pragma unroll
    for (int d = MAX_DIMS - 1; d>=0; --d){
        if (d < ndim){
            int64_t coord = temp % new_shape[d];
            temp /= new_shape[d];
            src_offset += coord * new_strides[d];
        }
    }
    out_ptr[idx] = src_ptr[src_offset];

}


void cuda_transpose(void* out_ptr, const void* src_ptr, uint64_t numel, int ndim, const int64_t* new_shape, const int64_t* new_strides, DType dtype){

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        // Pasar los metadatos (shape y strides) a la GPU
        int64_t *d_new_shape = nullptr, *d_permuted_strides = nullptr;
        size_t meta_bytes = ndim * sizeof(int64_t);

        cudaMalloc(&d_new_shape, meta_bytes);
        cudaMalloc(&d_permuted_strides, meta_bytes);

        cudaMemcpy(d_new_shape, new_shape, meta_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_permuted_strides, new_strides, meta_bytes, cudaMemcpyHostToDevice);

        // Configuración de ejecución
        int threads_per_block = 256;
        int blocks = (numel + threads_per_block - 1) / threads_per_block;

        transpose_kernel<scalar_t><<<blocks, threads_per_block>>>(
            static_cast<scalar_t*>(out_ptr),
            static_cast<const scalar_t*>(src_ptr),
            numel,
            ndim,
            d_new_shape,
            d_permuted_strides
        );

        cudaDeviceSynchronize();

        // Liberar metadatos temporales
        cudaFree(d_new_shape);
        cudaFree(d_permuted_strides);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Error en transpose_kernel: ") + cudaGetErrorString(err));
        }
    });
}

template<typename scalar_t>
__global__ void randn_kernel(scalar_t* __restrict__ out_ptr, const float* __restrict__ rand_buffer, uint64_t numel, float mean, float std){
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel){
        float val = rand_buffer[idx] * std + mean;
        out_ptr[idx] = static_cast<scalar_t>(val);
    }
}


void cuda_randn(void* out_ptr, uint64_t numel, float mean, float std, DType dtype) {
    if (numel == 0) return;

    // cuRAND requiere una cantidad par de elementos para distribución normal
    size_t curand_size = (numel % 2 == 0) ? numel : numel + 1;

    // 1. Buffer temporal de floats para cuRAND
    float* d_rand_buffer = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_rand_buffer, curand_size * sizeof(float));
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error("Error al reservar memoria temporal para cuRAND: " + std::string(cudaGetErrorString(cuda_err)));
    }

    // 2. Crear generador cuRAND
    curandGenerator_t gen;
    curandStatus_t curand_status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        cudaFree(d_rand_buffer);
        throw std::runtime_error("Error al crear generador cuRAND");
    }

    // Semilla pseudoaleatoria
    unsigned long long seed = std::random_device{}();
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    // 3. Generar N(0, 1) en buffer temporal
    curand_status = curandGenerateNormal(gen, d_rand_buffer, curand_size, 0.0f, 1.0f);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        curandDestroyGenerator(gen);
        cudaFree(d_rand_buffer);
        throw std::runtime_error("Error al generar números en cuRAND");
    }

    // 4. Escalar y castear a dtype final
    DISPATCH_ALL_TYPES(dtype, scalar_t, ({
        int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);

        randn_kernel<scalar_t><<<blocks, threads>>>(
            static_cast<scalar_t*>(out_ptr),
            d_rand_buffer,
            numel,
            mean,
            std
        );
    }));

    cudaDeviceSynchronize();

    // 5. Limpieza
    curandDestroyGenerator(gen);
    cudaFree(d_rand_buffer);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en scale_and_cast_kernel: " + std::string(cudaGetErrorString(err)));
    }
}