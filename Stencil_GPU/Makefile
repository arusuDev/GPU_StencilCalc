OBJS=Stencil_3D1GPU_128_1024 Stencil_3DMultiGPU_128_1024

all:$(OBJS)
Stencil_3D1GPU_128_1024:Stencil_3D1GPU.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
Stencil_3DMultiGPU_128_1024:Stencil_3DMultiGPU.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
