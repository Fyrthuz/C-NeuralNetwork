#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

struct LossResult {
    Tensor loss;
    Tensor grad;
};


LossResult cross_entropy(Tensor output, Tensor ground_truth, Device device, DType dtype);

// Variante que devuelve el gradiente respecto a las PROBABILIDADES
// (dL/dp = -Y / (P * B), 0 donde P < eps); el valor de la loss es identico.
// Se usa junto con la capa SoftMax: su backward convierte dL/dp en dL/dz.
LossResult cross_entropy_probs(Tensor output, Tensor ground_truth, Device device, DType dtype);

#endif