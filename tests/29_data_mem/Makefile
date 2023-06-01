TEST=test29_data_mem

all: $(TEST) kernel.openmp.so 

include ../Makefile.tests

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

kernel.nvopenmp.so: kernel.cl.openmp.c
	$(NVCC) -g -shared -I. ${INCLUDE} ${EXT_INCLUDE} --compiler-options -fPIC -o $@ $^

clean:
	rm -f kernel.ptx kernel.hip kernel.openmp.so test29_data_mem
