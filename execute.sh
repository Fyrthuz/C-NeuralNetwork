nvcc -std=c++17 -O2 main.cpp lib/tensor.cpp lib/ops.cpp lib/linear.cpp lib/loss.cpp lib/kernel_operations.cu lib/ops.cu lib/loss.cu -o programa.out -lcudart -lcurand
./programa.out

nvcc -std=c++17 -O2 main2.cpp lib/tensor.cpp lib/ops.cpp lib/linear.cpp lib/loss.cpp lib/kernel_operations.cu lib/ops.cu lib/loss.cu -o programa.out -lcudart -lcurand
./programa.out
