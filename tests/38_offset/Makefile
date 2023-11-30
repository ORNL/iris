TEST=test38_offset
CPP=1

all: $(TEST) kernel.ptx kernel.openmp.so kernel.hip

include ../Makefile.tests

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

kernel.nvopenmp.so: kernel.openmp.c
	$(NVCC) -g -shared -I. ${INCLUDE} ${EXT_INCLUDE} --compiler-options -fPIC -o $@ $^

clean:
	rm -f vecadd vecadd-iris vecadd-m kernel.ptx kernel.openmp.so kernel.hip
