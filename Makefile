OBJS=Stencil_P2P_128_128
all:$(OBJS)
Stencil_P2P_128_128:Stencil_P2P.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
