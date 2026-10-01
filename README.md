# C-NeuralNetwork

Red neuronal profunda implementada **desde cero en C++17**, sin ninguna librería externa
de deep learning (ni BLAS, ni Eigen, ni frameworks tipo PyTorch/TensorFlow).

El proyecto tiene dos partes:

1. Un **motor de tensores** propio (tipo NumPy/PyTorch) con broadcasting, `matmul`,
   activaciones y semántica completa de C++ (Rule of Five).
2. Un **perceptrón multicapa (MLP)** entrenado con backpropagation manual sobre **MNIST**.

---

## Arquitectura

```
784 (píxeles)  ->  Linear(128)  ->  ReLU
                ->  Linear(64)   ->  ReLU
                ->  Linear(10)   ->  Softmax  ->  Cross-Entropy
```

- Inicialización **Kaiming/He** en los pesos, `bias` en ceros.
- Activaciones: ReLU (ocultas), Softmax (salida).
- Loss: **cross-entropy** categórico con el gradiente fusionado `(P - Y) / batch_size`.
- Optimizador: **SGD** con learning rate configurable.

---

## Estructura del proyecto

```
.
├── main.cpp            # Suite de pruebas del motor de tensores (26 tests)
├── main2.cpp           # Entrenamiento y evaluación del MLP en MNIST
├── execute.sh          # Script de compilación + ejecución
├── lib/
│   ├── tensor.h/.cpp   # Clase Tensor: memoria, shape, strides, dtypes, semántica de movimiento
│   ├── ops.h/.cpp      # add, sub, mult, div, matmul, relu, softmax + broadcasting
│   ├── linear.h/.cpp   # LinearLayer, ReLU, SoftMax (forward + backward)
│   ├── loss.h/.cpp     # cross_entropy (devuelve loss y gradiente)
│   ├── optim.h         # SGD: zero_grad() y step()
│   └── dataset.h       # Carga de MNIST en CSV y extracción de minilotes
└── data/
    └── mnist-main/
        ├── mnist_train.csv/mnist_train.csv   # 60,000 muestras
        └── mnist_test.csv/mnist_test.csv     # 10,000 muestras
```

---

## Requisitos

- **g++** (o clang++) con soporte de **C++17**
- Dataset MNIST en CSV ya incluido en `data/`

No hace falta instalar nada más: solo la biblioteca estándar de C++.

---

## Compilación y ejecución

### Suite de pruebas (motor de tensores)

```bash
g++ -std=c++17 main.cpp lib/tensor.cpp lib/ops.cpp -o programa.out
./programa.out
```

Imprime el resultado de los 26 tests (shape, strides, dtypes, broadcasting,
excepciones) y termina con `TODAS LAS PRUEBAS FINALIZARON CON EXITO!`.

### Entrenamiento en MNIST

```bash
g++ -std=c++17 main2.cpp lib/tensor.cpp lib/ops.cpp lib/linear.cpp lib/loss.cpp -o programa.out
./programa.out
```

> **Nota:** `lib/loss.cpp` es obligatorio en este comando, porque `main2.cpp` usa
> `cross_entropy()`. Si lo omites, el enlazado falla con
> `undefined reference to 'cross_entropy(...)'`.

O simplemente:

```bash
chmod +x execute.sh
./execute.sh
```

---

## Configuración del entrenamiento

Los hiperparámetros están al final de `main2.cpp` (`main()`):

```cpp
SGD optimizer(0.08f);        // Learning rate
const int64_t batch_size = 64;
const int epochs = 15;
const std::string train_path = "data/mnist-main/mnist_train.csv/mnist_train.csv";
const std::string test_path  = "data/mnist-main/mnist_test.csv/mnist_test.csv";
```

### Salida esperada

```
==========================================
    INICIANDO PIPELINE DE MNIST
==========================================

[OK] Muestras cargadas: 60000

==========================================
       ENTRENAMIENTO (15 EPOCHS)
==========================================
Epoca [ 1/15] | Loss Promedio: 0.4213
Epoca [ 2/15] | Loss Promedio: 0.1874
...

==========================================
           EVALUACION FINAL
==========================================
Precision en Entrenamiento: 99.xx%
Precision en Test Set:      98.xx%
==========================================
```

La loss debería decrecer de forma monótona durante las 15 épocas. El script imprime
el accuracy en porcentajes sobre el set de entrenamiento y el de test.

---

## Detalles de implementación

### Motor de tensores (`lib/tensor.h`)

La clase `Tensor` es un wrapper de memoria con metadatos, al estilo de PyTorch:

- **Shape** multidimensional con `int64_t` y **strides** calculados automáticamente.
- **Dtypes**：`F32`, `F64`, `I32`, `I64`.
- **Devices**：`CPU` y `CUDA` (declarado; `CUDA` todavía no está implementado).
- **Semántica de movimiento** completa (constructor por movimiento, asignación por
  movimiento) para evitar copias innecesarias.
- **Factory methods**：`zeros`, `ones`, `arange`, `randn`.
- **Utilidades**：`clone()`, `flatten()`, `transpose()`.
- **Impresión** anidada según la dimensionalidad, con `operator<<`.

### Dispatch de tipos (macro `DISPATCH_ALL_TYPES`)

El "kernel" de cada operación se genera una vez por tipo y se compila como
`switch` sobre el `DType`. Así el bucle interno trabaja con `scalar_t` concreto
(sin coste de boxing) y el mismo código sirve para `float`, `double` o enteros:

```cpp
DISPATCH_ALL_TYPES(t.dtype(), scalar_t, {
    scalar_t* ptr = t.data_ptr<scalar_t>();
    // ... kernel ...
});
```

### Broadcasting (`lib/ops.cpp`)

`add`, `sub`, `mult` y `div` soportan broadcasting tipo NumPy (`[2,3] + [3]`,
`[2,3] - [2,1]`, etc.). Se calculan los *strides* de salida expandiendo con
`stride = 0` las dimensiones que se repiten, y se valida la compatibilidad
lanzando excepción si no lo es.

### `matmul`

Soporta las formas `[N, K] x [K, M] -> [N, M]`, con broadcasting opcional de lote.
Valida la compatibilidad de dimensiones internas y lanza excepción ante shapes
inválidos.

### Backpropagation

Cada capa cachea lo que necesita para el paso backward:

- `LinearLayer` guarda `input_cache` y calcula `d_weights = Xᵀ · dZ` y
  `d_bias = Σ dZ` (acumulado sobre el batch).
- `ReLU` guarda la entrada y enmascara el gradiente con `(x > 0)`.
- `cross_entropy` devuelve **loss y gradiente** en un solo recorrido
  (`LossResult`), evitando una pasada extra.
- `SGD::zero_grad()` limpia los gradientes con `memset` antes de cada iteración.

---

## Estado del proyecto

Implementado:

- [x] Motor de tensores con broadcasting y dtypes múltiples
- [x] Suite de 26 tests del motor de tensores
- [x] Capas `Linear`, `ReLU`, `SoftMax` con forward y backward
- [x] `cross_entropy` con gradiente
- [x] Optimizador SGD
- [x] Entrenamiento end-to-end en MNIST

Pendiente / ideas:

- [ ] Backpropagation para `SoftMax` como capa propia (ahora se integra en la loss)
- [ ] Shuffle de los datos y batches por época (hoy el orden es fijo)
- [ ] `Adam` / otros optimizadores
- [ ] Capas convolutivas para comparar con una CNN clásica
- [ ] Soporte real de GPU (el enum `Device::CUDA` ya existe)
- [ ] Unit tests separados con gtest / CMake en vez del `main` actual
- [ ] Guardar y cargar pesos del modelo entrenado

---

## Notas

- El dataset se carga **completo en memoria** (60,000 × 784 floats ≈ 188 MB), por lo
  que los minilotes se copian con `memcpy` desde el tensor del dataset.
- `evaluate_accuracy` solo procesa los batches completos (`count / batch_size`);
  las muestras sobrantes del último batch no se evalúan.
- `execute.sh` reconstruye y ejecuta ambos binarios en el mismo `programa.out`.
