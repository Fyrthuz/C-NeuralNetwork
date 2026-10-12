#include "kernel_operations.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>

// ------------------- element-wise binario mismo shape -------------------
template<typename scalar_t, typename Op>
__global__ void binary_same_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, uint64_t n, Op op)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}


// Functors templateados por scalar_t: preservan el tipo exacto (F32/F64/I32/I64)
// sin conversiones implícitas a float. Se pasan al kernel como Op=AddOp<scalar_t>
// desde dentro de DISPATCH_ALL_TYPES.
template<typename scalar_t>
struct AddOp  { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a + b; } };

template<typename scalar_t>
struct SubOp  { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a - b; } };

template<typename scalar_t>
struct MultOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a * b; } };

template<typename scalar_t>
struct DivOp  { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a / b; } };

// ------------------- ReLU -------------------
template<typename scalar_t>
__global__ void relu_kernel(scalar_t* out, const scalar_t* in, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > static_cast<scalar_t>(0) ? in[i] : static_cast<scalar_t>(0);
}


// ------------------- element-wise binario broadcasting -------------------
template<typename scalar_t, typename Op>
__global__ void binary_broadcast_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, uint64_t n, int ndim, const int64_t* __restrict__ shape, const int64_t* __restrict__ stride_a, const int64_t* __restrict__ stride_b, Op op)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t off_a = 0, off_b = 0, tmp = i;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t c = tmp % shape[d];
        tmp /= shape[d];
        off_a += c * stride_a[d];
        off_b += c * stride_b[d];
    }
    out[i] = op(a[off_a], b[off_b]);
}


template<template <typename> class Opt>
void cuda_binary_same(void* out, const void* a, const void* b, uint64_t n, DType dtype){
    if (n == 0) return;  // sin elementos no se lanza kernel (grid de 0 bloques es inválido)

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        binary_same_kernel<scalar_t, Opt<scalar_t>><<<blocks, threads>>>(static_cast<scalar_t*>(out), static_cast<const scalar_t*>(a), static_cast<const scalar_t*>(b), n, Opt<scalar_t>{});
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_binary_same: " + std::string(cudaGetErrorString(err)));
    }
}


void cuda_add_same(void* out, const void* a, const void* b, uint64_t n, DType dtype){
    cuda_binary_same<AddOp>(out, a, b, n, dtype);
}


void cuda_sub_same(void* out, const void* a, const void* b, uint64_t n, DType dtype){
    cuda_binary_same<SubOp>(out, a, b, n, dtype);
}


void cuda_mult_same(void* out, const void* a, const void* b, uint64_t n, DType dtype){
    cuda_binary_same<MultOp>(out, a, b, n, dtype);
}


void cuda_div_same(void* out, const void* a, const void* b, uint64_t n, DType dtype){
    cuda_binary_same<DivOp>(out, a, b, n, dtype);
}

// Operadores binarios con broadcast
template <template <typename> class Opt>
void cuda_binary_broadcast(void* out, const void* a, const void* b,
                           uint64_t n, int ndim,
                           const int64_t* shape,
                           const int64_t* stride_a,
                           const int64_t* stride_b,
                           DType dtype){
    
    if (n == 0) return;  // sin elementos no se lanza kernel

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        const size_t meta_bytes = static_cast<size_t>(ndim) * sizeof(int64_t);

        // Un solo malloc para shape | stride_a | stride_b (3 * ndim int64)
        int64_t* d_meta = nullptr;
        cudaError_t err = cudaMalloc(&d_meta, 3 * meta_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Error en cuda_binary_broadcast (malloc): ") + cudaGetErrorString(err));
        }

        cudaMemcpy(d_meta,            shape,    meta_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_meta + ndim,     stride_a, meta_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_meta + 2 * ndim, stride_b, meta_bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);

        binary_broadcast_kernel<scalar_t, Opt<scalar_t>><<<blocks, threads>>>(
            static_cast<scalar_t*>(out),
            static_cast<const scalar_t*>(a),
            static_cast<const scalar_t*>(b),
            n, ndim,
            d_meta, d_meta + ndim, d_meta + 2 * ndim,
            Opt<scalar_t>{}
        );

        err = cudaGetLastError();
        cudaFree(d_meta);

        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Error en cuda_binary_broadcast: ") + cudaGetErrorString(err));
        }
    });
}


void cuda_add_broadcast(void* out, const void* a, const void* b,uint64_t n, int ndim,const int64_t* shape,const int64_t* stride_a,const int64_t* stride_b, DType dtype) {
    cuda_binary_broadcast<AddOp>(out, a, b, n, ndim, shape, stride_a, stride_b, dtype);
}


void cuda_sub_broadcast(void* out, const void* a, const void* b, uint64_t n, int ndim, const int64_t* shape, const int64_t* stride_a, const int64_t* stride_b, DType dtype) {
    cuda_binary_broadcast<SubOp>(out, a, b, n, ndim, shape, stride_a, stride_b, dtype);
}


void cuda_mult_broadcast(void* out, const void* a, const void* b, uint64_t n, int ndim, const int64_t* shape, const int64_t* stride_a, const int64_t* stride_b, DType dtype) {
    cuda_binary_broadcast<MultOp>(out, a, b, n, ndim, shape, stride_a, stride_b, dtype);
}


void cuda_div_broadcast(void* out, const void* a, const void* b, uint64_t n, int ndim, const int64_t* shape, const int64_t* stride_a, const int64_t* stride_b, DType dtype) {
    cuda_binary_broadcast<DivOp>(out, a, b, n, ndim, shape, stride_a, stride_b, dtype);
}


// ------------------- Lanzador de ReLU -------------------
void cuda_relu(void* out, const void* in, uint64_t n, DType dtype){
    if (n == 0) return;  // sin elementos no se lanza kernel

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        relu_kernel<scalar_t><<<blocks, threads>>>(static_cast<scalar_t*>(out), static_cast<const scalar_t*>(in), n);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_relu: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- Matmul 2D -------------------
// Un hilo por elemento (i, j) del output [M, N]; acumula los K productos en
// orden k = 0..K-1 (mismo orden que la version CPU -> mismos resultados).
// El loop de batches vive en ops.cpp, que lanza este kernel una vez por batch
// con los punteros ya desplazados.
template<typename scalar_t>
__global__ void matmul_2d_kernel(scalar_t* __restrict__ out,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ B,
                                 int64_t M, int64_t K, int64_t N,
                                 int64_t sAM, int64_t sAK,
                                 int64_t sBK, int64_t sBN){
    int64_t i = blockIdx.y;                              // fila
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;   // columna
    if (i >= M || j >= N) return;

    scalar_t sum = 0;
    for (int64_t k = 0; k < K; ++k) {
        sum += A[i * sAM + k * sAK] * B[k * sBK + j * sBN];
    }
    out[i * N + j] = sum;
}


void cuda_matmul_2d(void* out, const void* a, const void* b,
                    int64_t M, int64_t K, int64_t N,
                    int64_t stride_aM, int64_t stride_aK,
                    int64_t stride_bK, int64_t stride_bN, DType dtype){
    if (M == 0 || N == 0) return;  // sin filas/columnas no hay nada que calcular

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        // grid: x = columnas (bloques), y = filas
        dim3 blocks(static_cast<unsigned>((N + threads - 1) / threads),
                    static_cast<unsigned>(M));

        matmul_2d_kernel<scalar_t><<<blocks, threads>>>(
            static_cast<scalar_t*>(out),
            static_cast<const scalar_t*>(a),
            static_cast<const scalar_t*>(b),
            M, K, N, stride_aM, stride_aK, stride_bK, stride_bN);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_matmul_2d: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- Softmax (un bloque por fila) -------------------
// Cada bloque normaliza una fila de C elementos: reduce el maximo (estabilidad
// numerica), reduce la suma de exp(x - max) y divide. Mismo algoritmo que la
// version CPU.
template<typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ in, int64_t C){
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const scalar_t* in_row  = in  + row * C;
    scalar_t*       out_row = out + row * C;

    __shared__ scalar_t smax[256];
    __shared__ scalar_t ssum[256];

    // 1) Maximo de la fila (C >= 1 garantizado por el lanzador)
    scalar_t local_max = in_row[0];
    for (int64_t j = tid; j < C; j += nthreads) {
        if (in_row[j] > local_max) local_max = in_row[j];
    }
    smax[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        __syncthreads();
    }
    const scalar_t max_val = smax[0];

    // 2) exp(x - max) y suma del denominador
    scalar_t local_sum = 0;
    for (int64_t j = tid; j < C; j += nthreads) {
        scalar_t e = static_cast<scalar_t>(exp(static_cast<double>(in_row[j] - max_val)));
        out_row[j] = e;
        local_sum += e;
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    const scalar_t sum_exp = ssum[0];

    // 3) Normalizar
    for (int64_t j = tid; j < C; j += nthreads) {
        out_row[j] /= sum_exp;
    }
}


void cuda_softmax(void* out, const void* in, int64_t outer, int64_t C, DType dtype){
    if (outer == 0 || C == 0) return;

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        const int threads = 256;  // == tamano de smax/ssum en el kernel
        softmax_kernel<scalar_t><<<static_cast<unsigned>(outer), threads>>>(
            static_cast<scalar_t*>(out),
            static_cast<const scalar_t*>(in),
            C);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_softmax: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- Backward de SoftMax -------------------
// dL/dz_j = p_j * (dL/dp_j - dot), con dot = sum_k dL/dp_k * p_k por fila.
// Un bloque por fila; la reduccion de `dot` va por shared memory.
template<typename scalar_t>
__global__ void softmax_backward_kernel(scalar_t* __restrict__ grad_in,
                                        const scalar_t* __restrict__ grad_out,
                                        const scalar_t* __restrict__ probs,
                                        int64_t C){
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const scalar_t* p_row  = probs    + row * C;
    const scalar_t* dp_row = grad_out + row * C;
    scalar_t*       o_row  = grad_in  + row * C;

    __shared__ scalar_t sdot[256];

    scalar_t local = static_cast<scalar_t>(0);
    for (int64_t j = tid; j < C; j += nthreads) {
        local += dp_row[j] * p_row[j];
    }
    sdot[tid] = local;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdot[tid] += sdot[tid + s];
        __syncthreads();
    }
    const scalar_t dot = sdot[0];

    for (int64_t j = tid; j < C; j += nthreads) {
        o_row[j] = p_row[j] * (dp_row[j] - dot);
    }
}


void cuda_softmax_backward(void* grad_in, const void* grad_out, const void* probs,
                           int64_t outer, int64_t C, DType dtype){
    if (outer == 0 || C == 0) return;

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        const int threads = 256;  // == tamano de sdot en el kernel
        softmax_backward_kernel<scalar_t><<<static_cast<unsigned>(outer), threads>>>(
            static_cast<scalar_t*>(grad_in),
            static_cast<const scalar_t*>(grad_out),
            static_cast<const scalar_t*>(probs),
            C);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_softmax_backward: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- Backward de Linear: acumular d_bias -------------------
// Una hebra por columna j; suma d_output[b, j] sobre todo el batch.
template<typename scalar_t>
__global__ void bias_accum_kernel(scalar_t* __restrict__ d_bias,
                                  const scalar_t* __restrict__ d_output,
                                  int64_t batch, int64_t out_features){
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_features) return;

    scalar_t sum = 0;
    for (int64_t b = 0; b < batch; ++b) {
        sum += d_output[b * out_features + j];
    }
    d_bias[j] = sum;
}


void cuda_bias_accum(void* d_bias, const void* d_output,
                     int64_t batch, int64_t out_features, DType dtype){
    if (out_features == 0 || batch == 0) return;  // sin batch, d_bias queda en ceros (ya lo puso LinearLayer::backward)

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((out_features + threads - 1) / threads);
        bias_accum_kernel<scalar_t><<<blocks, threads>>>(
            static_cast<scalar_t*>(d_bias),
            static_cast<const scalar_t*>(d_output),
            batch, out_features);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_bias_accum: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- Backward de ReLU -------------------
// grad_in[i] = (input_cache[i] > 0) ? grad_out[i] : 0
template<typename scalar_t>
__global__ void relu_backward_kernel(scalar_t* __restrict__ grad_in,
                                     const scalar_t* __restrict__ grad_out,
                                     const scalar_t* __restrict__ input_cache,
                                     uint64_t n){
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = (input_cache[i] > static_cast<scalar_t>(0)) ? grad_out[i] : static_cast<scalar_t>(0);
    }
}


void cuda_relu_backward(void* grad_in, const void* grad_out,
                        const void* input_cache, uint64_t n, DType dtype){
    if (n == 0) return;

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        relu_backward_kernel<scalar_t><<<blocks, threads>>>(
            static_cast<scalar_t*>(grad_in),
            static_cast<const scalar_t*>(grad_out),
            static_cast<const scalar_t*>(input_cache),
            n);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_relu_backward: " + std::string(cudaGetErrorString(err)));
    }
}


// ------------------- SGD: param -= lr * grad -------------------
template<typename scalar_t>
__global__ void sgd_step_kernel(scalar_t* __restrict__ param,
                                const scalar_t* __restrict__ grad,
                                scalar_t lr, uint64_t n){
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}


void cuda_sgd_step(void* param, const void* grad, float lr, uint64_t n, DType dtype){
    if (n == 0) return;

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        sgd_step_kernel<scalar_t><<<blocks, threads>>>(
            static_cast<scalar_t*>(param),
            static_cast<const scalar_t*>(grad),
            static_cast<scalar_t>(lr),
            n);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en cuda_sgd_step: " + std::string(cudaGetErrorString(err)));
    }
}

