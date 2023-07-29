include ../makefile_defs.mk

all: vecadd vecadd-iris kernel.ptx kernel.openmp.so kernel.hip

vecadd: vecadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-iris: vecadd-iris.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -lgomp -fPIC -shared -I. -o $@ $^

vecadd-sycl: vecadd-sycl.cpp
	$(CHARMSYCL)/bin/cscc --targets=cpu-c,nvidia-cuda -o $@ $^

DPCPP_HOME=/home/9bj/dpc++-workspace
vecadd-sycl-dpc++: vecadd-sycl.cpp
	$(DPCPP_HOME)/llvm/build/bin/clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -o $@ $^

#broken
LLVM_DIR=/home/9bj/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-12.1.0/llvm-13.0.1-2a5yipjwa6bco5vymttszyqauqzainuc
vecadd-opensycl-cuda: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_86" -L$(OPENSYCL)/lib/hipSYCL -lrt-backend-cuda -L$(OPENSYCL)/lib -lhipSYCL-rt -lopensycl-common -stdlib=libc++ -lstdc++ -lm  -o $@ $^
	#$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_86" --hipsycl-platform=cuda --hipsycl-gpu-arch=sm_62 -stdlib=libc++ -Wl,--rpath=$(LLVM_DIR)/lib/x87_64-unknown-linux-gnu -lstdc++ -lm -L$(OPENSYCL)/lib -L$(OPENSYCL)/lib/hipSYCL -lhipSYCL-rt -lrt-backend-cuda -L$(LLVM_DIR)/lib -o $@ $^

#broken
vecadd-opensycl-openmp: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -Wl,--rpath=/auto/software/swtree/ubuntu20.04/x86_64/gcc/12.1.0/lib64 -Wl,--dynamic-linker=/auto/software/swtree/ubuntu20.04/x86_64/gcc/12.1.0/lib64 -L/auto/software/swtree/ubuntu20.04/x86_64/gcc/12.1.0/lib64 -L/usr/lib -nostdlib -lpthread -pthread -lgomp -o $@ $^
	#$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -stdlib=libc++ -I$(OPENMP_PATH)/include -Wl,-rpath=$(OPENMP_PATH) -I/usr/lib/gcc/x86_64-linux-gnu/11/include -lgomp -o $@ $^

#broken
vecadd-computecpp: vecadd-sycl.cpp
	~/computecpp/bin/compute++ -I/home/9bj/computecpp/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/include -sycl-driver -sycl-target spir64 -o $@ $^


clean:
	rm -f vecadd vecadd-iris vecadd-m kernel.ptx kernel.openmp.so kernel.hip
