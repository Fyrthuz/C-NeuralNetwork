#include "tensor.h"
#include <stdexcept>
#include <cstring>
#include <utility>
#include <cmath>
#include <random>
#include <cstdlib>
#include <iostream>
#include <vector>


void* allocate_memory(uint64_t size_per_item, uint64_t items, Device device){
    void * data{nullptr};
    uint64_t total_bytes = size_per_item * items;
    if (items != 0 && total_bytes / items != size_per_item) {
        throw std::overflow_error("Desbordamiento al calcular el tamaño de memoria.");
    }

    if(device == Device::CPU){
        
        data = ::operator new(size_per_item * items, std::align_val_t{64});
    }else{
        /* IMPLEMENTACION PARA CUDA, PENDIENTE */
        throw std::runtime_error("CUDA no implementado todavía.");
    }
    return data;

}

void free_memory(void*& data, Device device){
    if (data) {
        if (device == Device::CPU) {
            ::operator delete(data, std::align_val_t{64});
        } else if (device == Device::CUDA) {
            /* IMPLEMENTACION PARA CUDA, PENDIENTE */
            throw std::runtime_error("CUDA no implementado todavía.");
        }
        data = nullptr;
    }
}


Tensor::Tensor(): data_(nullptr), shape_({}), strides_({}), numel_(0), dtype_(DType::F32), device_(Device::CPU)
{}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device) 
            : shape_(shape), dtype_(dtype), device_(device)
            {

                if (shape.empty()){
                    throw std::invalid_argument("Se necesita al menos una dimension para crear el tensor");
                }
                // Numero de elementos que va a tener n uestro tensor
                numel_ = 1;
                for (int64_t d : shape_){
                    if (d < 0){
                        throw std::invalid_argument("Las dimensiones no pueden ser negativas");
                    }
                    numel_ *= d;
                }

                // Calcular strides
                // Inicializamos nuestro strides
                strides_.resize(shape_.size(), 1);
                
                // Pasar de un elemento a otro tiene 1 stride, de una fila a otra tam_fila*1, y asi sucesivamente
                int64_t current_stride = 1;
                for(size_t i = shape_.size(); i > 0; i--){
                    strides_[i - 1] = current_stride;
                    current_stride *= shape_[i - 1];
                }

                // Reserva de memoria
                if (numel_ > 0){
                    data_ = allocate_memory(this->get_element_size(dtype), numel_, device_);
                }

            }

Tensor::~Tensor() {
    free_memory(data_, device_);
}



// --- Métodos de Copia y Movimiento pendientes de aprender ---

// Constructor de copia (Tensor b = a)
Tensor::Tensor(const Tensor& other) 
    : shape_(other.shape_), 
    strides_(other.strides_), 
    numel_(other.numel_), 
    dtype_(other.dtype_), 
    device_(other.device_),
    data_(nullptr)
{

    if(numel_ > 0 && other.data_ != nullptr){
        data_ = allocate_memory(this->get_element_size(this->dtype_), numel_, device_);
        if (device_ == Device::CPU){
            std::memcpy(this->data_, other.data_, this->numel_*this->get_element_size(this->dtype_));
        }else{
            /* IMPLEMENTACION PARA CUDA, PENDIENTE */
            throw std::runtime_error("CUDA no implementado todavía.");
        }
    }

}

// Asignacion por copia  (b = a)
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        Tensor temp(other); // Si esto falla, *this sigue intacto
        std::swap(data_, temp.data_);
        std::swap(shape_, temp.shape_);
        std::swap(strides_, temp.strides_);
        std::swap(numel_, temp.numel_);
        std::swap(dtype_, temp.dtype_);
        std::swap(device_, temp.device_);
    }
    return *this;
}

// Constructor por movimiento (T b = std::move(a);)
Tensor::Tensor(Tensor&& other) noexcept : 
numel_(std::exchange(other.numel_,0)),
shape_(std::exchange(other.shape_,{})),
strides_(std::exchange(other.strides_,{})),
data_(std::exchange(other.data_, nullptr)),
device_(other.device_),
dtype_(other.dtype_)
{}

// Asignacion por movimiento (b = std::move(a);)
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    // THis no puede ser igual a other
    if(this != &other){
        // Libero memoria
        if(data_ != nullptr){
            free_memory(data_, device_);
        }

        // Movemos todo de other para aqui
        data_    = other.data_;
        shape_   = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        numel_   = other.numel_;
        dtype_   = other.dtype_;
        device_  = other.device_;

        // Limpiamos other
        other.data_ = nullptr;
        other.numel_ = 0;
        other.strides_ = {};
        other.shape_ = {};
    }

    return *this;
}


// --- Métodos de Fábrica (Factory Methods) pendientes ---

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    if (t.numel() > 0){
        if(device == Device::CPU){
            std::memset(t.data(), 0, t.numel() * t.get_element_size(dtype));
        }else{
            throw std::runtime_error("CUDA no implementado todavía.");
        }
    }
    return t;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    
    Tensor t(shape, dtype, device);
    
    if (t.numel() == 0) return t;

    if(device == Device::CPU){
        DISPATCH_ALL_TYPES(dtype, scalar_t,{
            auto *ptr = static_cast<scalar_t *>(t.data_);
            for (uint64_t i = 0; i < t.numel(); i++){
                ptr[i] = static_cast<scalar_t>(1);
            }
        });
    } else{
        throw std::runtime_error("CUDA no implementado todavía.");
    }
    return t;
}

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step, DType dtype, Device device) {

    if (step == 0){
        throw std::invalid_argument("Step no puede ser 0");
    }
    
    int64_t diff = end - start;
    if ((diff > 0 && step < 0) || ( diff < 0 && step > 0)){
        throw std::invalid_argument("Pasa una combinacion buena de step, start, end");
    }

    // Calculamos tamaño del tensor
    int64_t size = static_cast<int64_t>((std::abs(diff)+std::abs(step)-1)/abs(step));

    // Instanciamos el tensor
    Tensor t({size}, dtype, device);

    if (size == 0){
        return t;
    }

    if (device == Device::CPU) {
        DISPATCH_ALL_TYPES(dtype, scalar_t, {
            auto *ptr = static_cast<scalar_t *>(t.data_);
            int64_t val = start;
            for(int64_t i = 0; i < size; ++i, val += step){
                ptr[i] = static_cast<scalar_t>(val);
            }
        });

    }else{
        throw std::runtime_error("CUDA no implementado todavía.");
    }
    return t;
}

// Para imprimir el tensor

void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << "\n[" << name << "]\n";
    } else {
        std::cout << "\n[Tensor]\n";
    }
    std::cout << *this << "\n";
}

static void print_scalar(std::ostream& os, const void * data, int64_t flat_idx, DType dtype){

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        os << static_cast<const scalar_t *>(data)[flat_idx];
    });
}



static void print_tensor_data(std::ostream& os, const Tensor& t, size_t dim, int64_t offset, int indent){

    // Caso base, llegamos a la ultima dimension y printeamos los escalares
    if (dim == t.shape().size()-1){
        os << "[";
        for (int64_t i = 0; i < t.shape()[dim]; ++i){
            print_scalar(os, t.data(), offset + i * t.strides()[dim], t.dtype());
            if (i+1 < t.shape()[dim]){
                os << ",";
            }
        }

        os << "]";
        return;
    }

    os << "[";
    for (int64_t i = 0; i < t.shape()[dim]; ++i){
        print_tensor_data(os, t, dim + 1, offset + i * t.strides()[dim], indent + 2);
        if (i + 1 < t.shape()[dim]) {
            os << ",\n" << std::string(indent, ' ');
        }
    }
    os << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "  - ndim:    " << t.ndim() << "\n"
       << "  - numel:   " << t.numel() << "\n"
       << "  - shape:   [";
    for (size_t i = 0; i < t.shape_.size(); ++i) {
        os << t.shape_[i] << (i + 1 == t.shape_.size() ? "" : ", ");
    }
    os << "]\n  - strides: [";
    for (size_t i = 0; i < t.strides_.size(); ++i) {
        os << t.strides_[i] << (i + 1 == t.strides_.size() ? "" : ", ");
    }
    os << "]\n  - data:    " << t.data_ << "\n";

    // Impresión de valores
    os << "  - values: ";
    if (t.data_ == nullptr || t.numel_ == 0) {
        os << "[]";
        return os;
    }

    if (t.device_ != Device::CPU) {
        os << "]";
        return os;
    }
    os << "\n" << std::string(8, ' ') ;

    if (t.ndim() == 0){
        print_scalar(os, t.data_, 0, t.dtype());
    }else{
        print_tensor_data(os, t, 0, 0, 8);
    }

    return os;
}

// Utilidades

Tensor Tensor::clone() const{
    return *this;
}


Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {


    int64_t ndim = static_cast<int64_t>(shape_.size());

    // Controlamos valores negativos
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::invalid_argument("Dimension fuera de rango para transponer");
    }

    // Controlamos que no se produzca el mismo tensor al final del proceso
    if (dim0 == dim1 || numel_ <= 1){
        return this->clone();
    }


    // Nueva shape y strides del tensor
    std::vector<int64_t> new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    std::vector<int64_t> new_strides = strides_;
    std::swap(new_strides[dim0], new_strides[dim1]);

    // Tensor resultado
    Tensor t = Tensor(new_shape, dtype_, device_);

    // Odometro de coordenadas y offsets
    std::vector<uint64_t> coords(ndim, 0);
    int64_t src_offset = 0;

    if (device_ == Device::CPU){

        DISPATCH_ALL_TYPES(this->dtype_, scalar_t, {
            // Datos destino y origen
            const auto* src_ptr = static_cast<scalar_t*>(data_);
            auto* set_ptr = static_cast<scalar_t*>(t.data_);

            for (uint64_t i = 0; i < t.numel_; i++){
                set_ptr[i] = src_ptr[src_offset];

                for (int64_t d = ndim -1; d >= 0; --d){
                    coords[d] += 1;
                    src_offset += new_strides[d];

                    if (coords[d] < new_shape[d]){
                        break;
                    }

                    src_offset -= (coords[d] * new_strides[d]);
                    coords[d] = 0;
                }
            }
        });

    }else{
        throw std::runtime_error("Version CUDA no implementada");
    }


    return t;
}

Tensor Tensor::flatten() const{

    int64_t d = numel_;
    // Our new tensor is going to have shape (numel, )
    Tensor t = Tensor({d}, dtype_, device_);

    // Copiamos data en bloque porque es contiguo
    std::memcpy(t.data_, data_, this->numel_*this->get_element_size(this->dtype_));
    return t;
}

// Inicializacion de pesos
Tensor Tensor::randn(const std::vector<int64_t>& shape, float mean, float std, DType dtype, Device device) {
    if (device != Device::CPU) {
        throw std::invalid_argument("CUDA no soportado");
    }

    Tensor t(shape, dtype, device);

    // Generador aleatorio estándar en C++
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);

    DISPATCH_ALL_TYPES(dtype, scalar_t, {
        scalar_t* ptr = t.data_ptr<scalar_t>();
        for (uint64_t i = 0; i < t.numel(); ++i) {
            ptr[i] = static_cast<scalar_t>(dist(gen));
        }
    });

    return t;
}