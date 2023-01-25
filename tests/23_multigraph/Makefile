TEST=test23_multigraph

all: $(TEST) kernel.openmp.so 

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.openmp.so: kernel.cl.openmp.c
	gcc -std=c99 -O3 -fopenmp -fPIC $(CFLAGS) -shared -I. -o $@ $^

run: $(TEST) kernel.ptx
	./$(TEST) 4096 6 5
