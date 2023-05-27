TEST=test18_record

all: $(TEST) kernel.openmp.so 

include ../Makefile.tests

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	gcc -g -fopenmp -fPIC -shared -I. ${CFLAGS} -o $@ $^

cleanjson:
	rm -f *.json
