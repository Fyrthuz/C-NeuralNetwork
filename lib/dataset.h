#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include "tensor.h"

struct MNISTDataset {
    Tensor images;  // Shape: [N, 784] normalizado [0.0, 1.0]
    Tensor targets; // Shape: [N, 10] en formato One-Hot
    int64_t count = 0;

    // Carga el archivo CSV (ignora cabeceras de texto si existen)
    static MNISTDataset load_csv(const std::string& filepath, int64_t max_samples = -1) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("No se pudo abrir el archivo CSV: " + filepath);
        }

        std::vector<float> pixel_buffer;
        std::vector<float> target_buffer;
        std::string line;

        std::cout << "[INFO] Leyendo dataset desde: " << filepath << "...\n";

        int64_t loaded = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string token;

            // 1. Leer el label (primera columna)
            if (!std::getline(ss, token, ',')) continue;

            // Saltar fila si es la cabecera (e.g. "label,pixel0,...")
            if (token == "label" || token == "Label") continue;

            int label = std::stoi(token);
            if (label < 0 || label > 9) {
                throw std::runtime_error("Etiqueta fuera de rango [0-9]: " + std::to_string(label));
            }

            // Preparar el vector One-Hot (10 clases)
            for (int c = 0; c < 10; ++c) {
                target_buffer.push_back((c == label) ? 1.0f : 0.0f);
            }

            // 2. Leer los 784 píxeles restantes
            int pixel_count = 0;
            while (std::getline(ss, token, ',')) {
                float px = std::stof(token) / 255.0f; // Normalizar a [0.0, 1.0]
                pixel_buffer.push_back(px);
                pixel_count++;
            }

            if (pixel_count != 784) {
                throw std::runtime_error("Fila con cantidad de pixeles invalida: " + std::to_string(pixel_count));
            }

            loaded++;
            if (max_samples > 0 && loaded >= max_samples) {
                break;
            }
        }

        std::cout << "[OK] Muestras cargadas: " << loaded << "\n";

        // 3. Empaquetar los buffers en tensores
        MNISTDataset ds;
        ds.count = loaded;
        ds.images = Tensor({loaded, 784}, DType::F32, Device::CPU);
        ds.targets = Tensor({loaded, 10}, DType::F32, Device::CPU);

        std::memcpy(ds.images.data_ptr<float>(), pixel_buffer.data(), pixel_buffer.size() * sizeof(float));
        std::memcpy(ds.targets.data_ptr<float>(), target_buffer.data(), target_buffer.size() * sizeof(float));

        return ds;
    }

    // Extrae un slice continuo para formar un minilote [batch_size, 784] y [batch_size, 10]
    void get_batch(int64_t start_idx, int64_t batch_size, Tensor& out_x, Tensor& out_y) const {
        if (start_idx + batch_size > count) {
            throw std::out_of_range("El batch excede los limites del dataset");
        }

        out_x = Tensor({batch_size, 784}, DType::F32, Device::CPU);
        out_y = Tensor({batch_size, 10}, DType::F32, Device::CPU);

        const float* src_x = images.data_ptr<float>() + (start_idx * 784);
        const float* src_y = targets.data_ptr<float>() + (start_idx * 10);

        std::memcpy(out_x.data_ptr<float>(), src_x, batch_size * 784 * sizeof(float));
        std::memcpy(out_y.data_ptr<float>(), src_y, batch_size * 10 * sizeof(float));
    }
};

#endif // DATASET_H