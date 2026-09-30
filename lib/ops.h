#ifndef OPS_H
#define OPS_H
#include "tensor.h"

Tensor add(Tensor t, Tensor p);
Tensor sub(Tensor t, Tensor p);
Tensor mult(Tensor t, Tensor p);
Tensor div(Tensor t, Tensor p);
Tensor matmul(Tensor t, Tensor p);
Tensor relu(const Tensor& t);
Tensor softmax(const Tensor& t, int64_t dim=-1);

#endif