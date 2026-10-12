#include "kernel_operations.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>


// =================================================================
// cross_entropy en CUDA: dos kernels independientes
//   1. ce_grad_kernel:        grad[i] = (probs[i] - target[i]) * b_inv
//   2. ce_loss_partials_kernel + ce_loss_finalize_kernel:
//      sumas parciales por bloque (reduccion en shared memory) y un bloque
//      final que las suma y escribe loss_out[0] = -suma * b_inv.
//      Sin atomicAdd(double) porque no esta disponible para arquitecturas
//      < sm_60 (y con los flags por defecto de nvcc no compila).
// =================================================================


// ------------------- Gradiente: (P - Y) / B -------------------
template<typename scalar_t>
__global__ void ce_grad_kernel(scalar_t* __restrict__ grad,
                               const scalar_t* __restrict__ probs,
                               const scalar_t* __restrict__ target,
                               uint64_t total, scalar_t b_inv){
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        grad[i] = (probs[i] - target[i]) * b_inv;
    }
}


void cuda_cross_entropy_grad(void* grad_out, const void* probs, const void* target,
                             uint64_t total, float b_inv, DType dtype){
    if (total == 0) return;

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        ce_grad_kernel<scalar_t><<<blocks, threads>>>(
            static_cast<scalar_t*>(grad_out),
            static_cast<const scalar_t*>(probs),
            static_cast<const scalar_t*>(target),
            total,
            static_cast<scalar_t>(b_inv));
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_cross_entropy_grad: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- Loss: suma parcial por bloque -------------------
template<typename scalar_t>
__global__ void ce_loss_partials_kernel(scalar_t* __restrict__ partials,
                                        const scalar_t* __restrict__ probs,
                                        const scalar_t* __restrict__ target,
                                        uint64_t total, scalar_t eps){
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t local = static_cast<scalar_t>(0);
    if (i < total && target[i] > static_cast<scalar_t>(0)) {
        scalar_t p = (probs[i] < eps) ? eps : probs[i];
        local = target[i] * static_cast<scalar_t>(log(static_cast<double>(p)));
    }

    // Reduccion del bloque en shared memory
    __shared__ scalar_t sdata[256];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
}


// ------------------- Loss: finalizacion (un solo bloque) -------------------
template<typename scalar_t>
__global__ void ce_loss_finalize_kernel(scalar_t* __restrict__ loss_out,
                                        const scalar_t* __restrict__ partials,
                                        int64_t n_partials, scalar_t b_inv){
    __shared__ scalar_t sdata[256];

    scalar_t local = static_cast<scalar_t>(0);
    for (int64_t i = threadIdx.x; i < n_partials; i += blockDim.x) {
        local += partials[i];
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        loss_out[0] = -sdata[0] * b_inv;
    }
}


void cuda_cross_entropy_loss(void* loss_out, const void* probs, const void* target,
                             uint64_t total, float eps, float b_inv, DType dtype){
    if (total == 0) return;

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);

        // Buffer temporal con una suma parcial por bloque
        scalar_t* d_partials = nullptr;
        cudaError_t err = cudaMalloc(&d_partials, static_cast<size_t>(blocks) * sizeof(scalar_t));
        if (err != cudaSuccess) {
            throw std::runtime_error("Error en cuda_cross_entropy_loss (malloc): " + std::string(cudaGetErrorString(err)));
        }

        ce_loss_partials_kernel<scalar_t><<<blocks, threads>>>(
            d_partials,
            static_cast<const scalar_t*>(probs),
            static_cast<const scalar_t*>(target),
            total,
            static_cast<scalar_t>(eps));

        ce_loss_finalize_kernel<scalar_t><<<1, threads>>>(
            static_cast<scalar_t*>(loss_out),
            d_partials,
            blocks,
            static_cast<scalar_t>(b_inv));

        err = cudaGetLastError();
        cudaFree(d_partials);

        if (err != cudaSuccess) {
            throw std::runtime_error("Error en cuda_cross_entropy_loss: " + std::string(cudaGetErrorString(err)));
        }
    });
}
