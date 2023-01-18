TEST=test23_multigraph

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.openmp.so: kernel.cl.openmp.c
	gcc -std=c99 -O3 -fopenmp -fPIC -shared -I. -o $@ $^

run: $(TEST) kernel.ptx
	./$(TEST) 4096 6 5
