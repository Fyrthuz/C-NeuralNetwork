#ifndef KERNEL_OPERATIONS_H
#define KERNEL_OPERATIONS_H
#pragma once
#include <cstdint>
#include "tensor.h" // donde tengas definido tu enum DType

// Declaraciones de funciones "lanzadoras" para la GPU
void cuda_arange(void* out_ptr, int64_t start, int64_t step, int64_t size, DType dtype);

void cuda_fill(void* out_ptr, double value, int64_t size, DType dtype);

void cuda_transpose(void* out_ptr, const void* src_ptr, uint64_t numel, int ndim, const int64_t* new_shape, const int64_t* new_strides, DType dtype);

void cuda_randn(void* out_ptr, uint64_t numel, float mean, float std, DType dtype);


void cuda_add_same(void* out, const void* a, const void* b, uint64_t n, DType dtype);
void cuda_sub_same(void* out, const void* a, const void* b, uint64_t n, DType dtype);
void cuda_mult_same(void* out, const void* a, const void* b, uint64_t n, DType dtype);
void cuda_div_same(void* out, const void* a, const void* b, uint64_t n, DType dtype);

// === element-wise binario con broadcasting ===
void cuda_add_broadcast(void* out, const void* a, const void* b,
                        uint64_t n, int ndim,
                        const int64_t* shape,
                        const int64_t* stride_a,
                        const int64_t* stride_b, DType dtype);

void cuda_sub_broadcast(void* out, const void* a, const void* b,
                        uint64_t n, int ndim,
                        const int64_t* shape,
                        const int64_t* stride_a,
                        const int64_t* stride_b, DType dtype);

void cuda_mult_broadcast(void* out, const void* a, const void* b,
                        uint64_t n, int ndim,
                        const int64_t* shape,
                        const int64_t* stride_a,
                        const int64_t* stride_b, DType dtype);

void cuda_div_broadcast(void* out, const void* a, const void* b,
                        uint64_t n, int ndim,
                        const int64_t* shape,
                        const int64_t* stride_a,
                        const int64_t* stride_b, DType dtype);

// === unarias ===
void cuda_relu(void* out, const void* in,  uint64_t n, DType dtype);

void cuda_softmax(void* out, const void* in,  int64_t outer, int64_t C, DType dtype);

void cuda_softmax_backward(void* grad_in, const void* grad_out, const void* probs,
                           int64_t outer, int64_t C, DType dtype);

// === matmul ===
void cuda_matmul_2d(void* out, const void* a, const void* b,
                    int64_t M, int64_t K, int64_t N,
                    int64_t stride_aM, int64_t stride_aK,
                    int64_t stride_bK, int64_t stride_bN, DType dtype);
// (broadcast de batch se hace en CPU loop que lanza cuda_matmul_2d por batch)

// === bias accumulate (lineal backward) ===
void cuda_bias_accum(void* d_bias, const void* d_output,
                    int64_t batch, int64_t out_features, DType dtype);

// === relu backward ===
void cuda_relu_backward(void* grad_in, const void* grad_out,
                        const void* input_cache, uint64_t n, DType dtype);

// === SGD step: param -= lr * grad ===
void cuda_sgd_step(void* param, const void* grad, float lr, uint64_t n, DType dtype);

// === cross_entropy ===
void cuda_cross_entropy_grad(void* grad_out,
                             const void* probs,
                             const void* target,
                             uint64_t total, float b_inv, DType dtype);

// Gradiente respecto a las PROBABILIDADES: -Y/(P*B), 0 donde P < eps.
void cuda_cross_entropy_probs_grad(void* grad_out,
                                   const void* probs,
                                   const void* target,
                                   uint64_t total, float eps, float b_inv, DType dtype);

// Implementada en loss.cu sin atómicos (atomicAdd(double) no existe < sm_60):
// sumas parciales por bloque + un bloque final que escribe -suma * b_inv.
void cuda_cross_entropy_loss(void* loss_out,
                             const void* probs,
                             const void* target,
                             uint64_t total, float eps, float b_inv, DType dtype);

#endif