TEST=test33_graph_cpp
CPP=1
include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.openmp.so: kernel.cl.openmp.c
	gcc -g -fopenmp -fPIC -shared -I. -o $@ $^
