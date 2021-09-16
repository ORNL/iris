IRIS=$(HOME)/.local
CC=gcc
CXX=g++
FORTRAN=gfortran
CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
CXXFLAGS=-I$(IRIS)/include/ -O3
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -liris -lpthread -ldl

all: saxpy-c saxpy-cpp saxpy-f90 kernel.openmp.so

saxpy-c: saxpy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-cpp: saxpy.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-f90: saxpy.f90
	$(FORTRAN) $(FFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.spv: kernel.cl
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -flto -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

kernel.hexagon.so: kernel.hexagon.c
	$(CC) -g -fPIC -shared -I. -o $@ $^

kernel.poly.so: kernel.cl.poly.c
	$(CC) -g -fPIC -shared -I. -o $@ $^

clean:
	rm -f saxpy-c saxpy-cpp saxpy-f90
