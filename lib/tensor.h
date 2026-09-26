#ifndef TENSOR_H
#define TENSOR_H

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
        // Default escalar vacio
        Tensor(); 

        // Constructor normal
        Tensor(const std::vector<uint64_t>& shape,
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
        static Tensor zeros(const std::vector<uint64_t>& shape, DType dtype=DType::F32);
        static Tensor ones(const std::vector<uint64_t>& shape, DType dtype=DType::F32);
        static Tensor arange(int64_t start, int64_t end, int64_t step = 1);


        // Getters
        const std::vector<uint64_t>& shape() const { return shape_ ;};
        const std::vector<uint64_t>& strides() const { return strides_ ;};
        uint64_t numel() const {return numel_;};
        uint64_t ndim() const {return shape_.size();};
        DType dtype() const {return dtype_;};
        Device device() const {return device_;};

        // Acceso al buffer
        void* data() { return data_; };
        const void* data() const { return data_; };


        // Acceso tipado conveniente
        template <typename T> 
        T* data_ptr() { return static_cast<T*>(data_); };


        template <typename T>
        const T* data_ptr() const { return static_cast<const T*>(data_); };

    
    private:
        void* data_{nullptr}; // Puntero a los datos crudos
        std::vector<uint64_t> shape_; // Shape del tensor
        std::vector<uint64_t> strides_; // Strides del tensor
        uint64_t numel_{0};  // Numero total de elementos
        DType dtype_{DType::F32}; // Tipo de dato
        Device device_{Device::CPU}; // Device por defecto



};


#endif