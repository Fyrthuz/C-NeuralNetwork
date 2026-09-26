#include "tensor.h"
#include <stdexcept>


Tensor::Tensor(): data_(nullptr), shape_({}), strides_({}), numel_(0), dtype_(DType::F32), device_(Device::CPU)
{}

Tensor::Tensor(const std::vector<uint64_t>& shape, DType dtype, Device device) 
            : shape_(shape), dtype_(dtype), device_(device)
            {
                // Numero de elementos que va a tener n uestro tensor
                numel_ = 1;
                for (uint64_t d : shape_){
                    numel_ *= d;
                }
                if (shape_.empty()) numel_ = 0; // Si no hay dimensiones

                // Calcular strides
                // Inicializamos nuestro strides
                strides_.resize(shape_.size(), 1);
                
                // Pasar de un elemento a otro tiene 1 stride, de una fila a otra tam_fila*1, y asi sucesivamente
                uint64_t current_stride = 1;
                for(size_t i = shape_.size(); i > 0; i--){
                    strides_[i - 1] = current_stride;
                    current_stride *= shape_[i - 1];
                }

                // Numero de bytes
                uint64_t bytes_per_item = 0;
                switch (dtype_)
                {
                    case DType::F32:
                    case DType::I32:
                        bytes_per_item = 4;
                        break;
                    case DType::F64:
                    case DType::I64:
                        bytes_per_item = 8;
                        break;
                }

                // Reserva de memoria
                if (numel_ > 0){
                    if(device_ == Device::CPU){
                        // Pide la memoria cruda al SO, en vez de hacer malloc. Es un puntero crudo
                        data_ = ::operator new(numel_ * bytes_per_item);
                    }else{
                        /* IMPLEMENTACION PARA CUDA, PENDIENTE */
                        throw std::runtime_error("CUDA no implementado todavía.");
                    }
                }

            }

Tensor::~Tensor() {
    if (data_) {
        if (device_ == Device::CPU) {
            ::operator delete(data_);
        } else if (device_ == Device::CUDA) {
            // Aquí llamarías a cudaFree(data_);
        }
        data_ = nullptr;
    }
}



// --- Métodos de Copia y Movimiento pendientes de aprender ---

Tensor::Tensor(const Tensor& other) {
    throw std::runtime_error("Constructor de copia no implementado todavia.");
}

Tensor::Tensor(Tensor&& other) noexcept {
    throw std::runtime_error("Constructor de movimiento no implementado todavia.");
}

Tensor& Tensor::operator=(const Tensor& other) {
    throw std::runtime_error("Operador de asignacion por copia no implementado todavia.");
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    throw std::runtime_error("Operador de asignacion por movimiento no implementado todavia.");
}


// --- Métodos de Fábrica (Factory Methods) pendientes ---

Tensor Tensor::zeros(const std::vector<uint64_t>& shape, DType dtype) {
    throw std::runtime_error("Metodo Tensor::zeros no implementado todavia.");
}

Tensor Tensor::ones(const std::vector<uint64_t>& shape, DType dtype) {
    throw std::runtime_error("Metodo Tensor::ones no implementado todavia.");
}

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step) {
    throw std::runtime_error("Metodo Tensor::arange no implementado todavia.");
}
