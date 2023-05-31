CPP=1

TEST=test27_deadlock2

all: $(TEST) kernel.openmp.so 

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.openmp.so: kernel.openmp.c kernel.openmp.h
	gcc -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^
