#ifndef OPTIM_H
#define OPTIM_H

#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include "tensor.h"
#include "linear.h"
#include "kernel_operations.h"

class SGD {
public:
    // Punteros a pares de (parámetro, gradiente)
    struct ParamGradPair {
        Tensor* param;
        Tensor* grad;
    };

    std::vector<ParamGradPair> params;
    float lr;

    explicit SGD(float learning_rate = 0.01f) : lr(learning_rate) {}

    // Registra una capa lineal para que el optimizador gestione sus pesos y sesgos
    void register_layer(LinearLayer& layer) {
        params.push_back({&layer.weights, &layer.d_weights});
        params.push_back({&layer.bias, &layer.d_bias});
    }

    // Pone todos los gradientes a 0 antes de una nueva pasada backward
    void zero_grad() {
        for (auto& pair : params) {
            Tensor* g = pair.grad;
            if (g->data_ptr<void>() == nullptr || g->numel() == 0) {
                continue;
            }

            size_t bytes = g->numel() * g->get_element_size(g->dtype());

            if (g->device() == Device::CUDA) {
                // cudaMemset en vez de std::memset: el puntero vive en la GPU
                cudaError_t err = cudaMemset(g->data(), 0, bytes);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("SGD::zero_grad (CUDA): ") + cudaGetErrorString(err));
                }
            } else {
                std::memset(g->data(), 0, bytes);
            }
        }
    }

    // Actualiza: Param = Param - lr * Grad
    void step() {
        for (auto& pair : params) {
            Tensor* p = pair.param;
            Tensor* g = pair.grad;

            if (p->numel() == 0 || g->numel() == 0) continue;
            if (p->shape() != g->shape()) {
                throw std::runtime_error("SGD: El gradiente no coincide en dimensiones con el parametro");
            }

            if (p->device() == Device::CUDA) {
                cuda_sgd_step(p->data(), g->data(), lr, p->numel(), p->dtype());
            } else {
                DISPATCH_ALL_TYPES(p->dtype(), scalar_t, {
                    scalar_t* p_ptr = p->data_ptr<scalar_t>();
                    const scalar_t* g_ptr = g->data_ptr<scalar_t>();
                    uint64_t total = p->numel();
                    scalar_t rate = static_cast<scalar_t>(lr);

                    for (uint64_t i = 0; i < total; ++i) {
                        p_ptr[i] -= rate * g_ptr[i];
                    }
                });
            }
        }
    }
};

#endif // OPTIM_H
