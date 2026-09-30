#include "tensor.h"
#include <cmath>
Tensor cross_entropy(Tensor output, Tensor ground_truth, Device device, DType dtype){
    
    if (output.shape() != ground_truth.shape()) {
        throw std::invalid_argument("output y ground_truth deben tener la misma forma");
    }
    if (output.dtype() != ground_truth.dtype()) {
        throw std::invalid_argument("Los tensores deben tener el mismo tipo de dato");
    }
    
    Tensor result = Tensor({1}, dtype, device);
    int64_t batch_size = output.shape()[0];
    const float eps = 1e-7f; // Evitar log(0)
    
    if (device == Device::CPU){
        DISPATCH_ALL_TYPES(dtype, scalar_t,{
            float value = 0.0f;
            const scalar_t* out_ptr = output.data_ptr<scalar_t>();
            const scalar_t* gt_ptr  = ground_truth.data_ptr<scalar_t>();
            scalar_t* res_ptr       = result.data_ptr<scalar_t>();

            scalar_t total_loss = 0;
            uint64_t total_elements = output.numel();

            for (uint64_t i = 0; i < total_elements; ++i) {
                // Solo acumula cuando ground_truth es distinto de 0
                if (gt_ptr[i] > static_cast<scalar_t>(0)) {
                    scalar_t prob = (out_ptr[i] < static_cast<scalar_t>(eps)) ? static_cast<scalar_t>(eps) : out_ptr[i];
                    total_loss += gt_ptr[i] * std::log(prob);
                }
            }

            res_ptr[0] = -total_loss / static_cast<scalar_t>(batch_size);
        });
    }else{
        throw std::invalid_argument("CUDA no soportado");
    }

    return result;

}