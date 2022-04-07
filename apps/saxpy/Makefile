include ../makefile_defs.mk

all: saxpy-c saxpy-cpp kernel.openmp.so

saxpy-c: saxpy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-cpp: saxpy.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

saxpy-f90: saxpy.f90
	$(FORTRAN) $(FFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.spv: kernel.cl
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -flto -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

kernel.hexagon.so: kernel.hexagon.c
	$(CC) $(CFLAGS) -g -fPIC -shared -I. -o $@ $^

kernel.poly.so: kernel.cl.poly.c
	$(CC) $(CFLAGS) -g -fPIC -shared -I. -o $@ $^

clean:
	rm -f saxpy-c saxpy-cpp saxpy-f90 kernel.openmp.so
