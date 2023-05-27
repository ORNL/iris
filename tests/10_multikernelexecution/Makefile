TEST=test10_multikernelexecution
all: $(TEST) kernel.ptx kernel.openmp.so kernel.hip

include ../Makefile.tests

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	gcc -g -fopenmp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

