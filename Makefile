OBJS=Stencil_P2P
all:$(OBJS)
Stencil_P2P:Stencil_P2P.cu
	nvcc -arch=sm_60 $? -O3 -Xcompiler -fopenmp -o $@
