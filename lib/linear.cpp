#include "linear.h"
#include "ops.h"
#include "tensor.h"
#include "kernel_operations.h"
#include <cmath>

LinearLayer::LinearLayer(
    int64_t in_features,
    int64_t out_features,
    Device device,
    DType dtype
) {
    // Inicialización Kaiming / He
    float std = std::sqrt(2.0f / static_cast<float>(in_features));


    weights = Tensor::randn({in_features, out_features}, 0.0f, std, dtype, device);
    bias = Tensor::zeros({1, out_features}, dtype, device);

    d_weights = Tensor::zeros({in_features, out_features}, dtype, device);
    d_bias = Tensor::zeros({1, out_features}, dtype, device);

}

Tensor LinearLayer::forward(const Tensor& input) {
    Tensor output = matmul(input, weights);
    input_cache = input;
    output = add(output, bias);
    return output;
}

Tensor LinearLayer::backward(const Tensor& d_output){
    d_weights = matmul(input_cache.transpose(1,0), d_output);
    // Reiniciar d_bias a ceros antes de acumular
    d_bias = Tensor::zeros(bias.shape(), bias.dtype(), bias.device());
    int64_t batch_size = d_output.shape()[0];
    int64_t out_features = d_output.shape()[1];
    
    if (d_output.device() == Device::CUDA){
        cuda_bias_accum(d_bias.data(), d_output.data(), batch_size, out_features, d_output.dtype());
    } else {
        DISPATCH_ALL_TYPES(d_output.dtype(), scalar_t, {
            const scalar_t* dout_ptr = d_output.data_ptr<scalar_t>();
            scalar_t* dbias_ptr = d_bias.data_ptr<scalar_t>();

            for (int64_t b = 0; b < batch_size; ++b) {
                for (int64_t j = 0; j < out_features; ++j) {
                    dbias_ptr[j] += dout_ptr[b * out_features + j];
                }
            }
        });
    }

    Tensor d_in = matmul(d_output, weights.transpose(1, 0));

    return d_in;

}



Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    return relu(input);
}


Tensor ReLU::backward(const Tensor& d_output){

    Tensor d_in = Tensor(input_cache.shape(), input_cache.dtype(), input_cache.device());

    if (d_output.device() == Device::CUDA){
        cuda_relu_backward(d_in.data(), d_output.data(), input_cache.data(), d_output.numel(), d_output.dtype());
    } else {
        DISPATCH_ALL_TYPES(d_output.dtype(), scalar_t, {
            const scalar_t* ptr = d_output.data_ptr<scalar_t>();
            const scalar_t* in_ptr = input_cache.data_ptr<scalar_t>();
            scalar_t* result_ptr = d_in.data_ptr<scalar_t>();
            const scalar_t zero = static_cast<scalar_t>(0);
            for(uint64_t i = 0; i < d_output.numel(); ++i){
                result_ptr[i] = (in_ptr[i] > zero) ? ptr[i] : zero;
            }
        });
    }

    return d_in;

}


Tensor SoftMax::forward(const Tensor& input) {
    Tensor out = softmax(input);
    output_cache = out;   // el backward necesita las probabilidades
    return out;
}


Tensor SoftMax::backward(const Tensor& d_output){
    // Jacobiano del softmax aplicado a dL/dp, por filas:
    //   dL/dz_j = p_j * (dL/dp_j - sum_k dL/dp_k * p_k)
    const Tensor& probs = output_cache;

    Tensor d_in(probs.shape(), probs.dtype(), probs.device());

    int64_t ndim = static_cast<int64_t>(probs.shape().size());
    int64_t C = probs.shape()[ndim - 1];
    int64_t outer = static_cast<int64_t>(probs.numel()) / C;

    if (d_output.device() == Device::CUDA){
        cuda_softmax_backward(d_in.data(), d_output.data(), probs.data(), outer, C, d_output.dtype());
    } else {
        DISPATCH_ALL_TYPES(d_output.dtype(), scalar_t, {
            const scalar_t* dp = d_output.data_ptr<scalar_t>();
            const scalar_t* p  = probs.data_ptr<scalar_t>();
            scalar_t* o = d_in.data_ptr<scalar_t>();

            for (int64_t row = 0; row < outer; ++row){
                const scalar_t* p_row  = p  + row * C;
                const scalar_t* dp_row = dp + row * C;
                scalar_t* o_row = o + row * C;

                // dot = sum_k dL/dp_k * p_k
                scalar_t dot = static_cast<scalar_t>(0);
                for (int64_t j = 0; j < C; ++j) {
                    dot += dp_row[j] * p_row[j];
                }

                for (int64_t j = 0; j < C; ++j) {
                    o_row[j] = p_row[j] * (dp_row[j] - dot);
                }
            }
        });
    }

    return d_in;
}