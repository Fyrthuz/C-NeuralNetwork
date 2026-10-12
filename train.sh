#!/usr/bin/env bash
# =====================================================================
# train.sh — Compila y entrena el MLP configurable sobre MNIST (main4).
#
# Uso:
#   ./train.sh [opciones de main4]
#
# Ejemplos:
#   ./train.sh                                       # GPU, F32, 784-128-64-10
#   ./train.sh --device cpu --limit 10000            # CPU con 10k muestras
#   ./train.sh --dtype f64 --epochs 5                # doble precision
#   ./train.sh --layers 784,256,128,64,10 --lr 0.05  # otra arquitectura
#   ./train.sh --help                                # todas las opciones
# =====================================================================
set -e
cd "$(dirname "$0")"

echo "[BUILD] Compilando main4 (nvcc)..."
nvcc -std=c++17 -O2 main4.cpp \
     lib/tensor.cpp lib/ops.cpp lib/linear.cpp lib/loss.cpp \
     lib/kernel_operations.cu lib/ops.cu lib/loss.cu \
     -o train_mnist.out -lcudart -lcurand

echo "[RUN] ./train_mnist.out $*"
echo ""
./train_mnist.out "$@"
