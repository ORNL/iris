include ../makefile_defs.mk

all: helloworld-c helloworld-cpp

helloworld-c: helloworld.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

helloworld-cpp: helloworld.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.spv: kernel.cl
	#clang -cc1 -finclude-default-header -triple spir $^ -O0 -emit-llvm-bc -o kernel.bc
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -flto -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	#hipcc --genco --targets gfx906 -o $@ $^
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	#$(CC) -O3 -qsmp -fPIC -shared -I. -o $@ $^
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

kernel.hexagon.so: kernel.hexagon.c
	$(CC) $(CFLAGS) -g -fPIC -shared -I. -o $@ $^

kernel.poly.so: kernel.cl.poly.c
	$(CC) $(CFLAGS) -g -fPIC -shared -I. -o $@ $^

clean:
	rm -f helloworld-c helloworld-cpp
