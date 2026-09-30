#include "tensor.h"
#include "loss.h"
#include <cmath>


LossResult cross_entropy(Tensor output, Tensor ground_truth, Device device, DType dtype){
    
    if (output.shape() != ground_truth.shape()) {
        throw std::invalid_argument("output y ground_truth deben tener la misma forma");
    }
    if (output.dtype() != ground_truth.dtype()) {
        throw std::invalid_argument("Los tensores deben tener el mismo tipo de dato");
    }
    
    // Tensor para el valor de perdida
    Tensor loss = Tensor({1}, dtype, device);
    // Tensor con la misma forma que output para el gradiente
    Tensor grad(output.shape(), dtype, device);

    int64_t batch_size = output.shape()[0];
    const float eps = 1e-7f; // Evitar log(0)
    
    if (device == Device::CPU){
        DISPATCH_ALL_TYPES(dtype, scalar_t,{
            float value = 0.0f;
            const scalar_t* out_ptr = output.data_ptr<scalar_t>();
            const scalar_t* gt_ptr  = ground_truth.data_ptr<scalar_t>();
            scalar_t* loss_ptr       = loss.data_ptr<scalar_t>();
            scalar_t* grad_ptr      = grad.data_ptr<scalar_t>();

            scalar_t total_loss = 0;
            uint64_t total_elements = output.numel();
            scalar_t b_inv = static_cast<scalar_t>(1) / static_cast<scalar_t>(batch_size);

            for (uint64_t i = 0; i < total_elements; ++i) {
                // A) Gradiente dL/dz: (P - Y) / Batch
                grad_ptr[i] = (out_ptr[i] - gt_ptr[i]) * b_inv;


                // B) Pérdida: -sum(Y * log(P))
                if (gt_ptr[i] > static_cast<scalar_t>(0)) {
                    scalar_t prob = (out_ptr[i] < static_cast<scalar_t>(eps)) 
                                    ? static_cast<scalar_t>(eps) 
                                    : out_ptr[i];
                    total_loss += gt_ptr[i] * std::log(prob);
                }
            }

            loss_ptr[0] = -total_loss * b_inv;
        });
    }else{
        throw std::invalid_argument("CUDA no soportado");
    }

    return {loss, grad};

}