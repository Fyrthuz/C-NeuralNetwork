#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class LinearLayer {
public:
    LinearLayer(
        int64_t in_features,
        int64_t out_features,
        Device device,
        DType dtype
    );

    Tensor forward(const Tensor& input);

private:
    Tensor weights;
    Tensor bias;
};


class ReLU{
   public:
    ReLU() = default;
    Tensor forward(const Tensor& input);

    private:
};


class SoftMax{
public:
    SoftMax() = default;
    Tensor forward(const Tensor& input);
private:
};

#endif