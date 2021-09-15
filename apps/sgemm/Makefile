CC=gcc
CXX=g++

all: sgemm kernel.openmp.so

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) -std=c99 -fopenmp -fPIC -shared -I. -o $@ $^

sgemm: sgemm.c
	$(CC) -O3 -std=c99 -o $@ $^ -liris -lpthread -ldl

clean:
	rm -f sgemm
