TEST=test33_graph_cpp
CPP=1
all: $(TEST) kernel.openmp.so kernel.ptx kernel.hip

include ../Makefile.tests

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	gcc -g -fopenmp -fPIC -shared -I. $(CFLAGS) -o $@ $^
