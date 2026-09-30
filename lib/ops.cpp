#include "tensor.h"
#include <stdexcept>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <limits>


struct BroadcastResult {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride_a;
    std::vector<int64_t> stride_b;
};

// Ejemplo (2, 4, 3) y (3) --> Habria que operar con (2, 4, 3) y (1, 1, 3) --> Rellenamos de derecha a izquierda los strides 
// y si es 1 el shape de la dimension  es como si no avanzara en el vector de datos
// Función genérica que opera sobre vectores
BroadcastResult get_broadcast_strides_raw(
    const std::vector<int64_t>& shape_a, const std::vector<int64_t>& strides_a,
    const std::vector<int64_t>& shape_b, const std::vector<int64_t>& strides_b) 
{
    int64_t ndim_a = shape_a.size();
    int64_t ndim_b = shape_b.size();
    int64_t max_dim = std::max(ndim_a, ndim_b);

    std::vector<int64_t> stride_res_a(max_dim, 0);
    std::vector<int64_t> stride_res_b(max_dim, 0);
    std::vector<int64_t> output_shape(max_dim, 0);

    for (int64_t i = 0; i < max_dim; ++i) {
        int64_t idx_a = ndim_a - i - 1;
        int64_t idx_b = ndim_b - i - 1;
        int64_t out_idx = max_dim - i - 1;

        int64_t dim_a = (idx_a >= 0) ? shape_a[idx_a] : 1;
        int64_t dim_b = (idx_b >= 0) ? shape_b[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument("Dimensiones no compatibles para broadcasting");
        }

        output_shape[out_idx] = std::max(dim_a, dim_b);

        stride_res_a[out_idx] = (dim_a == 1) ? 0 : strides_a[idx_a];
        stride_res_b[out_idx] = (dim_b == 1) ? 0 : strides_b[idx_b];
    }

    return {output_shape, stride_res_a, stride_res_b};
}


BroadcastResult get_broadcast_strides(const Tensor& t, const Tensor& p) {
    return get_broadcast_strides_raw(t.shape(), t.strides(), p.shape(), p.strides());
}

template<typename Op>
Tensor binary_operation(const Tensor& t, const Tensor& p, Op op){
    // Checkeos de device y de tisor& t, const Tensor& p, Op op){
    // Checkeos de device y de tipo de datos
    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }

    
    // Si tienen la misma shape no hace falta broadcasting
    if(t.shape() == p.shape()){
        Tensor result = Tensor(p.shape(), p.dtype(), p.device());
        if (t.device() == Device::CPU){
            DISPATCH_ALL_TYPES(p.dtype(), scalar_t, {
                scalar_t* result_ptr = result.data_ptr<scalar_t>();
                const scalar_t* v1 = t.data_ptr<scalar_t>();
                const scalar_t* v2 = p.data_ptr<scalar_t>();
                for (uint64_t i = 0; i < p.numel(); i++){
                    result_ptr[i] = op(v1[i], v2[i]);
                }
            });
        }else{
            throw std::invalid_argument("CUDA no soportado");
        }
        return result;
    }

    // Obtenemos la info necesario de shape, y strides para construir el tensor de retorno
    BroadcastResult br = get_broadcast_strides(t, p);

    
    // Instanciamos el tensor de resultado
    Tensor result = Tensor(br.shape, p.dtype(), p.device());

    uint64_t num_elementos_final = result.numel();
    std::vector<uint64_t> coord(br.shape.size(), 0);

    if (t.device() == Device::CPU){
        DISPATCH_ALL_TYPES(p.dtype(), scalar_t,{
            
            scalar_t* result_ptr = result.data_ptr<scalar_t>();
            
            const scalar_t* v1 = t.data_ptr<scalar_t>();
            const scalar_t* v2 = p.data_ptr<scalar_t>();

            int64_t offset_a = 0;
            int64_t offset_b = 0;
            for (int64_t i = 0; i < num_elementos_final; ++i){
                result_ptr[i] = op(v1[offset_a], v2[offset_b]);
                for (int64_t d = static_cast<int64_t>(result.shape().size()) - 1; d >= 0; --d ){
                    coord[d] += 1;
                    offset_a += br.stride_a[d];
                    offset_b += br.stride_b[d];
                    
                    // SI es menor operamos, si es mayor el odometro pasa a la sifuiente dimension, algoritmo similar a la transpuesta de un tensor
                    if (coord[d] < br.shape[d]){
                        break;
                    }

                    offset_a -= (coord[d] * br.stride_a[d]);
                    offset_b -= (coord[d] * br.stride_b[d]);
                    coord[d] = 0;
                }
            }
        });

    }else{
        throw std::invalid_argument("CUDA no implementado");
    }



    return result;
}


Tensor add(Tensor t, Tensor p){
    return binary_operation(t, p, [](auto a, auto b) -> auto {return a + b;} );
}


Tensor sub(Tensor t, Tensor p){
   return binary_operation(t, p, [](auto a, auto b) -> auto { return a - b;} );
}



Tensor mult(Tensor t, Tensor p){
    return binary_operation(t, p, [](auto a, auto b) -> auto {return a * b;} );
}

Tensor div(Tensor t, Tensor p){
    return binary_operation(t, p, [](auto a, auto b) -> auto {return a / b;} );
}



Tensor matmul(Tensor t, Tensor p){
    // Comprobaciones basicas
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }
    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }
    if (t.shape().size() < 2 || p.shape().size() < 2){
        throw std::invalid_argument("Shapes no validas para matmul");
    }

    uint64_t ndim_t = t.shape().size();
    uint64_t ndim_p = p.shape().size();

    // (M*K)*(K*N)= (M*N)
    int64_t M = t.shape()[ndim_t-2];
    int64_t K1 = t.shape()[ndim_t-1];
    int64_t K2 = p.shape()[ndim_p-2];
    int64_t N = p.shape()[ndim_p-1];

    if (K1 != K2){
        throw std::invalid_argument("Dimensiones no compatibles para matmul");
    }

    // Strides de las matrices 2D internas
    int64_t stride_t_M = t.strides()[ndim_t - 2];
    int64_t stride_t_K = t.strides()[ndim_t - 1];
    int64_t stride_p_K = p.strides()[ndim_p - 2];
    int64_t stride_p_N = p.strides()[ndim_p - 1];

    // Strides de las matrices 2D internas
    std::vector<int64_t> batch_dimensions_a(t.shape().begin(), t.shape().end() - 2);
    std::vector<int64_t> batch_dimensions_b(p.shape().begin(), p.shape().end() - 2);

    std::vector<int64_t> batch_strides_a(t.strides().begin(), t.strides().end() - 2);
    std::vector<int64_t> batch_strides_b(p.strides().begin(), p.strides().end() - 2);

    BroadcastResult br = get_broadcast_strides_raw(batch_dimensions_a, batch_strides_a, batch_dimensions_b, batch_strides_b);
    // Guardamos informacion relativa a los batches
    int64_t total_batches = std::accumulate(begin(br.shape), end(br.shape), 1, std::multiplies<int64_t>());
    int64_t batch_ndim = br.shape.size();
    
    // Metemos las dimensiones que va a tener de output
    std::vector<int64_t> out_shape = br.shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    // Creamos el tensor
    Tensor result = Tensor(out_shape, p.dtype(), p.device());

    // Hacemos los calculos
    if (t.device() == Device::CUDA){
        throw std::invalid_argument("CUDA no soportado");
    }else{
        DISPATCH_ALL_TYPES(result.dtype(), scalar_t, {
            scalar_t* result_ptr = result.data_ptr<scalar_t>();
            const scalar_t* v1 = t.data_ptr<scalar_t>();
            const scalar_t* v2 = p.data_ptr<scalar_t>();

            // Offset de los punteros y los odometros
            int64_t offset_batch_a = 0;
            int64_t offset_batch_b = 0;
            std::vector<int64_t> coord(batch_ndim, 0);
            int64_t matrix_size = M * N;
            
            // Iteramos por todo el batch
            for (int64_t b = 0; b < total_batches; ++b) {
                // Avanzamos el puntero al indice_batch*tamaño_matriz
                scalar_t* cur_result = result_ptr + (b * matrix_size);
                const scalar_t* cur_a = v1 + offset_batch_a;
                const scalar_t* cur_b = v2 + offset_batch_b;

                // Multiplicación 2D para el batch actual: (M x K) @ (K x N) -> (M x N)
                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        scalar_t sum = 0;
                        for (int64_t k = 0; k < K1; ++k) {
                            sum += cur_a[i * stride_t_M + k * stride_t_K] *
                                   cur_b[k * stride_p_K + j * stride_p_N];
                        }
                        cur_result[i * N + j] = sum;
                    }
                }

                // Odómetro para avanzar en las dimensiones de batch igual que el transponer y que antes
                for (int64_t d = batch_ndim - 1; d >= 0; --d) {
                    coord[d] += 1;
                    offset_batch_a += br.stride_a[d];
                    offset_batch_b += br.stride_b[d];

                    if (coord[d] < br.shape[d]) {
                        break;
                    }

                    offset_batch_a -= (coord[d] * br.stride_a[d]);
                    offset_batch_b -= (coord[d] * br.stride_b[d]);
                    coord[d] = 0;
                }
            }
        });
    }



    return result;

}


Tensor relu(const Tensor& t){
    Tensor result(t.shape(), t.dtype(), t.device());
    if (t.device() == Device::CPU){
        DISPATCH_ALL_TYPES(t.dtype(), scalar_t, {
            scalar_t* out_ptr = result.data_ptr<scalar_t>();
            const scalar_t* in_ptr = t.data_ptr<scalar_t>();
            for (uint64_t i = 0; i < t.numel(); ++i){
                out_ptr[i] = (in_ptr[i] > static_cast<scalar_t>(0)) ? in_ptr[i] : static_cast<scalar_t>(0);
            }
        })

    }else{
        throw std::invalid_argument("CUDA aun no soportado");
    }
    return result;
}


Tensor softmax(const Tensor& t, int64_t dim = -1){
    Tensor result(t.shape(), t.dtype(), t.device());
    if (t.device() == Device::CPU){
        int64_t ndim = t.shape().size();

        // Normalizar dimensión negativa
        if (dim < 0) dim += ndim;
        if (dim != ndim - 1) {
            throw std::invalid_argument("Por ahora softmax solo soporta la ultima dimension");
        }

        int64_t C = t.shape()[ndim - 1];          // Tamaño de la dimensión softmax (ej: num_clases)
        int64_t outer_size = t.numel() / C;      // Número total de vectores/filas a normalizar

        DISPATCH_ALL_TYPES(t.dtype(), scalar_t, {
            const scalar_t* in_ptr = t.data_ptr<scalar_t>();
            scalar_t* out_ptr = result.data_ptr<scalar_t>();

            for (int64_t row = 0; row < outer_size; ++row) {
                const scalar_t* in_row = in_ptr + (row * C);
                scalar_t* out_row = out_ptr + (row * C);

                // 1. Encontrar el valor máximo de la fila (estabilidad numérica)
                scalar_t max_val = in_row[0];
                for (int64_t j = 1; j < C; ++j) {
                    if (in_row[j] > max_val) max_val = in_row[j];
                }

                // 2. Exponenciar restando el máximo y calcular la suma del denominador
                scalar_t sum_exp = 0;
                for (int64_t j = 0; j < C; ++j) {
                    scalar_t exp_val = std::exp(in_row[j] - max_val);
                    out_row[j] = exp_val;
                    sum_exp += exp_val;
                }

                // 3. Normalizar dividiendo entre la suma total
                for (int64_t j = 0; j < C; ++j) {
                    out_row[j] /= sum_exp;
                }
            }
        });

    }else{
        throw std::invalid_argument("CUDA aun no soportado");
    }
    return result;
}



