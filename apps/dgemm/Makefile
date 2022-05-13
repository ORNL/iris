include ../makefile_defs.mk

all: dgemm kernel.openmp.so

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -g -std=c99 -fopenmp -fPIC -shared -I. -o $@ $^

dgemm: dgemm.c
	$(CC) $(CFLAGS) -g -O3 -std=c99 -o $@ $^ $(LDFLAGS)

clean:
	rm -f dgemm kernel.openmp.so
