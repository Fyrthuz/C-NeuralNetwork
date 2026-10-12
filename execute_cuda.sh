nvcc -std=c++17 main3.cpp lib/tensor.cpp lib/kernel_operations.cu -o programa_cuda.out -lcudart -lcurand
./programa_cuda.out