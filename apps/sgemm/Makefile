include ../makefile_defs.mk

all: sgemm kernel.openmp.so

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -g -std=c99 -fopenmp -fPIC -shared -I. -o $@ $^

sgemm: sgemm.c
	$(CC) $(CFLAGS) -g -O3 -std=c99 -o $@ $^ $(LDFLAGS)

clean:
	rm -f sgemm kernel.openmp.so
