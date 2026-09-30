#include "tensor.h"
#include <stdexcept>

Tensor add(Tensor t, Tensor p){
    // Comprobaciones basicas
    if (t.shape() != p.shape()){
        throw std::invalid_argument("Los tensores tienen que tener las mismas dimensiones");
    }
    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }

    Tensor result = Tensor(p.shape(), p.dtype(), p.device());

    if (t.device() == Device::CPU){
        DISPATCH_ALL_TYPES(p.dtype(), scalar_t, {
            scalar_t* result_ptr = static_cast<scalar_t *>(result.data());
            const scalar_t* v1 = static_cast<scalar_t *>(t.data());
            const scalar_t* v2 = static_cast<scalar_t *>(p.data());
            for (uint64_t i = 0; i < p.numel(); i++){
                result_ptr[i] = v1[i] + v2[i];
            }
        });
    }else{
        throw std::invalid_argument("CUDA no soportado");
    }

    return result;


}


Tensor sub(Tensor t, Tensor p){
    // Comprobaciones basicas
    if (t.shape() != p.shape()){
        throw std::invalid_argument("Los tensores tienen que tener las mismas dimensiones");
    }
    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }

    Tensor result = Tensor(p.shape(), p.dtype(), p.device());

    if (t.device() == Device::CPU){
        DISPATCH_ALL_TYPES(p.dtype(), scalar_t, {
            scalar_t* result_ptr = static_cast<scalar_t *>(result.data());
            const scalar_t* v1 = static_cast<scalar_t *>(t.data());
            const scalar_t* v2 = static_cast<scalar_t *>(p.data());
            for (uint64_t i = 0; i < p.numel(); i++){
                result_ptr[i] = v1[i] - v2[i];
            }
        });
    }else{
        throw std::invalid_argument("CUDA no soportado");
    }

    return result;
}



Tensor mult(Tensor t, Tensor p){
    // Comprobaciones basicas
    if (t.shape() != p.shape()){
        throw std::invalid_argument("Los tensores tienen que tener las mismas dimensiones");
    }
    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }

    Tensor result = Tensor(p.shape(), p.dtype(), p.device());

    if (t.device() == Device::CPU){
        DISPATCH_ALL_TYPES(p.dtype(), scalar_t, {
            scalar_t* result_ptr = static_cast<scalar_t *>(result.data());
            const scalar_t* v1 = static_cast<scalar_t *>(t.data());
            const scalar_t* v2 = static_cast<scalar_t *>(p.data());
            for (uint64_t i = 0; i < p.numel(); i++){
                result_ptr[i] = v1[i] * v2[i];
            }
        });
    }else{
        throw std::invalid_argument("CUDA no soportado");
    }

    return result;


}

Tensor div(Tensor t, Tensor p){
    // Comprobaciones basicas
    if (t.shape() != p.shape()){
        throw std::invalid_argument("Los tensores tienen que tener las mismas dimensiones");
    }
    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }

    Tensor result = Tensor(p.shape(), p.dtype(), p.device());

    if (t.device() == Device::CPU){
        DISPATCH_ALL_TYPES(p.dtype(), scalar_t, {
            scalar_t* result_ptr = static_cast<scalar_t *>(result.data());
            const scalar_t* v1 = static_cast<scalar_t *>(t.data());
            const scalar_t* v2 = static_cast<scalar_t *>(p.data());
            for (uint64_t i = 0; i < p.numel(); i++){
                result_ptr[i] = v1[i] / v2[i];
            }
        });
    }else{
        throw std::invalid_argument("CUDA no soportado");
    }

    return result;


}



Tensor matmul(Tensor t, Tensor p){
    // Comprobaciones basicas
    if (t.dtype() != p.dtype()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo tipo de dato");
    }

    if (t.shape().size() != 2 || p.shape().size() != 2) {
        throw std::invalid_argument("matmul actualmente solo soporta tensores 2D");
    }

    // (M*K)*(K*N)= (M*N)
    int64_t M = t.shape()[t.shape().size()-2];
    int64_t K1 = t.shape()[t.shape().size()-1];
    int64_t K2 = p.shape()[p.shape().size()-2];
    int64_t N = p.shape()[p.shape().size()-1];

    if (K1 != K2){
        throw std::invalid_argument("Dimensiones no compatibles para matmul");
    }
    Tensor result = Tensor({M,N}, p.dtype(), p.device());

    if (t.device() != p.device()){
        throw std::invalid_argument("Los tensores tienen que estar en el mismo device");
    }else{
        DISPATCH_ALL_TYPES(result.dtype(), scalar_t, {
            scalar_t* result_ptr = static_cast<scalar_t*>(result.data());

            const scalar_t* v1 = static_cast<const scalar_t*>(t.data());

            const scalar_t* v2 = static_cast<const scalar_t*>(p.data());

            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {

                    scalar_t sum = 0;

                    for (int64_t k = 0; k < K1; k++) {
                        sum += v1[i * K1 + k] *
                            v2[k * N + j];
                    }

                    result_ptr[i * N + j] = sum;
                }
            }


        });
    }



    return result;

}




