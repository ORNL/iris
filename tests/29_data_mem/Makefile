TEST=test29_data_mem

all: $(TEST) kernel.openmp.so  kernel.hip kernel.ptx kernel.nvopenmp.so

include ../Makefile.tests

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

ifeq ($(NVCC_TEST),)
kernel.nvopenmp.so: kernel.cl.openmp.c
	@echo "No NVCC compiler found"
else
kernel.nvopenmp.so: kernel.cl.openmp.c
	$(NVCC) -g -shared -I. ${INCLUDE} ${EXT_INCLUDE} --compiler-options -fPIC -o $@ $^
endif

clean:
	rm -f kernel.ptx kernel.hip kernel.openmp.so test29_data_mem
