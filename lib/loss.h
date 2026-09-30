#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

struct LossResult {
    Tensor loss;
    Tensor grad;
};


LossResult cross_entropy(Tensor output, Tensor ground_truth, Device device, DType dtype);

#endif