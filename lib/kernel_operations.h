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

#endif