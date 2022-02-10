IRIS_INSTALL_ROOT ?= $(HOME)/.local
IRIS=$(IRIS_INSTALL_ROOT)
CC=gcc
CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
LDFLAGS=-L$(IRIS)/lib -L$(IRIS)/lib64 -liris -lpthread -ldl

all: sgemm kernel.openmp.so

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -std=c99 -fopenmp -fPIC -shared -I. -o $@ $^

sgemm: sgemm.c
	$(CC) $(CFLAGS) -O3 -std=c99 -o $@ $^ $(LDFLAGS)

clean:
	rm -f sgemm
