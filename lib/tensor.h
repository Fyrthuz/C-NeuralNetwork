#ifndef TENSOR_H
#define TENSOR_H

#define DISPATCH_ALL_TYPES(dtype, TYPE_NAME, ...) \
    switch (dtype) { \
        case DType::F32: { using TYPE_NAME = float;   __VA_ARGS__; break; } \
        case DType::F64: { using TYPE_NAME = double;  __VA_ARGS__; break; } \
        case DType::I32: { using TYPE_NAME = int32_t; __VA_ARGS__; break; } \
        case DType::I64: { using TYPE_NAME = int64_t; __VA_ARGS__; break; } \
        default: throw std::runtime_error("DType no soportado");}


#include <vector>
#include <cstdint>
#include <ostream>


enum class DType {
    F32, // float
    F64, // double
    I32, // int32_t
    I64  // int64_t
};


enum class Device{
    CPU,
    CUDA
};



class Tensor{
    public:
        // Default
        Tensor(); 

        // Constructor normal
        Tensor(const std::vector<int64_t>& shape,
                DType dtype = DType::F32,
                Device device = Device::CPU);

        // Constructor "copia profunda"
        Tensor(const Tensor& other);

        // Constructor movimiento
        Tensor(Tensor&& other) noexcept;
        
        // Copia por asignacion
        Tensor& operator=(const Tensor& other);
        
        // Copia por movimiento
        Tensor& operator=(Tensor&& other) noexcept;
        
        // Destructor
        ~Tensor();


        // Factory methods
        static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::F32, Device device = Device::CPU);
        static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::F32, Device device = Device::CPU);
        static Tensor arange(int64_t start, int64_t end, int64_t step = 1, DType dtype = DType::F32, Device device = Device::CPU);
        inline static uint64_t get_element_size(DType dtype){
                switch (dtype)
                    {
                        case DType::F32: return sizeof(float);
                        case DType::I32: return sizeof(uint32_t);
                        case DType::F64: return sizeof(double);
                        case DType::I64: return sizeof(uint64_t);
                    }
                throw std::invalid_argument("Tipo Dtype no soportado o desconocido");
            }        
        
        // Utilidades
        Tensor clone() const;
        Tensor flatten() const;
        Tensor transpose(int64_t dim0, int64_t dim1) const;

        // Getters
        const std::vector<int64_t>& shape() const { return shape_ ;}
        const std::vector<int64_t>& strides() const { return strides_ ;}
        uint64_t numel() const {return numel_;}
        size_t ndim() const {return shape_.size();}
        DType dtype() const {return dtype_;}
        Device device() const {return device_;}

        // Acceso al buffer
        void* data() { return data_; }
        const void* data() const { return data_; }


        // Acceso tipado conveniente
        template <typename T> 
        T* data_ptr() { return static_cast<T*>(data_); }


        template <typename T>
        const T* data_ptr() const { return static_cast<const T*>(data_); }

        // Para imprimir el tensor
        // Método miembro directo
        void print(const std::string& name = "") const;

        // Sobrecarga del operador de inserción en stream (amiga de la clase)
        friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

    
    private:
        void* data_{nullptr}; // Puntero a los datos crudos
        std::vector<int64_t> shape_; // Shape del tensor
        std::vector<int64_t> strides_; // Strides del tensor
        uint64_t numel_{0};  // Numero total de elementos
        DType dtype_{DType::F32}; // Tipo de dato
        Device device_{Device::CPU}; // Device por defecto



};


#endif