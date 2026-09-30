#include "linear.h"
#include "ops.h"
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
}

Tensor LinearLayer::forward(const Tensor& input) {
    Tensor output = matmul(input, weights);
    output = add(output, bias);
    return output;
}



Tensor ReLU::forward(const Tensor& input) {
    return relu(input);
}



Tensor SoftMax::forward(const Tensor& input) {
    return softmax(input);
}