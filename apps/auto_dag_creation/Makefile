include ../makefile_defs.mk

all: auto_dag kernel.openmp.so kernel.ptx

auto_dag: auto_dag.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

auto_dag-cpp: auto_dag.c
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

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
	rm -f auto_dag kernel.openmp.so
