#include "linear.h"
#include "ops.h"

LinearLayer::LinearLayer(
    int64_t in_features,
    int64_t out_features,
    Device device,
    DType dtype
) {
    weights = Tensor({in_features, out_features}, dtype, device);
    bias = Tensor({1, out_features}, dtype, device);
}

Tensor LinearLayer::forward(const Tensor& input) {
    Tensor output = matmul(input, weights);

    return output;
}