include ../makefile_defs.mk

all: 2tasks kernel.openmp.so kernel.ptx

2tasks: 2tasks.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

clean:
	rm -f 2tasks kernel.openmp.so
