#include <iostream>
#include <vector>
#include "lib/tensor.h" // Apunta a la carpeta lib

int main() {
    std::cout << "--- Probando nuestra libreria de Tensores ---" << std::endl;

    // Creamos un shape de 2 filas y 3 columnas
    std::vector<uint64_t> mi_shape = {2, 3};

    // Inicializamos el tensor (llamará a tu constructor)
    Tensor t(mi_shape, DType::F32, Device::CPU);

    // Comprobamos los metadatos calculados
    std::cout << "Dimensiones (ndim): " << t.ndim() << std::endl;
    std::cout << "Numero de elementos (numel): " << t.numel() << std::endl;
    
    // Verificamos los strides calculados
    std::cout << "Strides: [";
    for(size_t i = 0; i < t.shape().size(); ++i) {
        std::cout << t.strides()[i] << (i == t.shape().size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // Validamos que se haya reservado la memoria
    if (t.data() != nullptr) {
        std::cout << "¡Memoria reservada con exito en la direccion: " << t.data() << "!" << std::endl;
    }

    return 0;
}
