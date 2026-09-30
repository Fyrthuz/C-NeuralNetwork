#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

Tensor cross_entropy(Tensor output, Tensor ground_truth, Device device, DType dtype);

#endif