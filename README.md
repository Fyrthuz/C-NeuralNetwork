# C-NeuralNetwork
g++ -std=c++17 main2.cpp lib/tensor.cpp lib/ops.cpp lib/linear.cpp -o programa.out

g++ -std=c++17 main.cpp lib/tensor.cpp lib/ops.cpp -o programa.out



# Tensor Engine — Estado actual y roadmap

## 1. Tensor

- [x] Clase `Tensor`
- [x] Shape
- [x] Numel
- [x] Strides
- [x] DType
- [x] Device
- [x] Gestión dinámica de memoria
- [x] Constructor
- [x] Destructor
- [x] Copy constructor
- [x] Copy assignment
- [x] Move constructor
- [x] Move assignment
- [x] `data()`
- [x] `clone()`
- [x] `flatten()`
- [x] `transpose()`
- [x] `zeros()`
- [x] `ones()`
- [x] `arange()`
- [x] Impresión del tensor

---

## 2. Operaciones básicas

### Element-wise

- [x] `add()`
- [x] `sub()`
- [x] `mult()`
- [x] `div()`

### Matrix operations

- [x] `matmul()` 2D
- [ ] Broadcasting
- [ ] Batched `matmul()`
- [ ] Broadcasting en `matmul()`

### CUDA

- [ ] `add()` CUDA
- [ ] `sub()` CUDA
- [ ] `mult()` CUDA
- [ ] `div()` CUDA
- [ ] `matmul()` CUDA
- [ ] Kernels CUDA optimizados

---

# 3. Linear Layer

## Estructura

- [x] Crear `LinearLayer`
- [x] `weights`
- [x] `bias`
- [x] Constructor
- [x] `forward()`
- [x] Usar `matmul()` dentro de `forward()`

Actualmente:

```text
input  [batch, in_features]
          @
weights [in_features, out_features]
          ↓
output [batch, out_features]