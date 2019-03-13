OBJS= Stencil_1D1GPU Stencil_1DMultiGPU Stencil_2DMultiGPU Stencil_3DMultiGPU Stencil_2DCPU Stencil_2D1GPU Stencil_3D1GPU
all:$(OBJS)
Stencil_3DMultiGPU:Stencil_3DMultiGPU.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
Stencil_3D1GPU:Stencil_3D1GPU.cu
	nvcc -arch=sm_60 $? -O3 -o $@
Stencil_2DMultiGPU:Stencil_2DMultiGPU.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
Stencil_2D1GPU:Stencil_2D1GPU.cu
	nvcc -arch=sm_60 $? -O3 -o $@
Stencil_2DCPU:Stencil_2DCPU.cpp
	g++ $? -o -O3 $@
Stencil_1D1GPU:Stencil_1D1GPU.cu
	nvcc -arch=sm_60 $? -O3 -o $@
Stencil_1DMultiGPU:Stencil_1DMultiGPU.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
	
