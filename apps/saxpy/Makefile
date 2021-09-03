BRISBANE=$(HOME)/.local
CC=xlc
CXX=xlC
#CC=gcc
#CXX=g++
FORTRAN=gfortran
CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
CXXFLAGS=-I$(IRIS)/include/ -O3
FFLAGS=-g -I$(IRIS)/include/brisbane
LDFLAGS=-L$(IRIS)/lib64 -lbrisbane -lpthread -ldl

all: saxpy saxpy-cpp saxpy-uvm

saxpy-seq: saxpy-seq.c
	$(CC) $(CFLAGS) -o $@ $^

saxpy: saxpy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-m: saxpy-m.c
	$(CC) -g $(CFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-mr: saxpy-mr.c
	$(CC) -g $(CFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-uvm: saxpy-uvm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-cpp: saxpy.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-f90: saxpy.f90
	$(FORTRAN) $(FFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.spv: kernel.cl
	#clang -cc1 -finclude-default-header -triple spir $^ -O0 -emit-llvm-bc -o kernel.bc
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -flto -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	#hipcc --genco --targets gfx906 -o $@ $^
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	#$(CC) -O3 -qsmp -fPIC -shared -I. -o $@ $^
	$(CC) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

kernel.hexagon.so: kernel.hexagon.c
	$(CC) -g -fPIC -shared -I. -o $@ $^

kernel.poly.so: kernel.cl.poly.c
	$(CC) -g -fPIC -shared -I. -o $@ $^

clean:
	rm -f saxpy saxpy-cpp saxpy-f90 saxpy-uvm
