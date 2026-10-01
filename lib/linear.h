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

    Tensor backward(const Tensor& d_output);
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    Tensor d_weights;
    Tensor d_bias;

private:
    
};


class ReLU{
public:
    ReLU() = default;
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& d_output);
    Tensor input_cache;

private:

};


class SoftMax{
public:
    SoftMax() = default;
    Tensor forward(const Tensor& input);
    
private:
};

#endif