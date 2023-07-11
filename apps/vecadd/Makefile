include ../makefile_defs.mk

all: vecadd vecadd-iris kernel.ptx kernel.openmp.so kernel.hip

vecadd: vecadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-iris: vecadd-iris.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

#data-memory optimization should remove 1 extra memory transfer
vecadd-iris-data-memory: vecadd-iris-data-memory.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -lgomp -fPIC -shared -I. -o $@ $^

vecadd-sycl: vecadd-sycl.cpp
	$(CHARMSYCL)/bin/cscc --targets=cpu-c,nvidia-cuda $(CXXFLAGS) -o $@ $^

#this optimization uses discard_write --like iris data-memory should avoid iris_task_h2d() being called.
vecadd-sycl-discard-write: vecadd-sycl-discard-write.cpp
	$(CHARMSYCL)/bin/cscc --targets=cpu-c,nvidia-cuda $(CXXFLAGS) -o $@ $^

DPCPP_HOME=/home/9bj/dpc++-workspace
vecadd-sycl-dpc++: vecadd-sycl.cpp
	$(DPCPP_HOME)/llvm/build/bin/clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -o $@ $^

vecadd-sycl-dpc++-discard-write: vecadd-sycl-discard-write.cpp
	$(DPCPP_HOME)/llvm/build/bin/clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -o $@ $^

vecadd-opensycl-openmp: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -pthread -lgomp -O3 -o $@ $^

vecadd-opensycl-openmp-discard-write: vecadd-sycl-discard-write.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -pthread -lgomp -O3 -o $@ $^


#broken
#LLVM_DIR=/home/9bj/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-12.1.0/llvm-13.0.1-2a5yipjwa6bco5vymttszyqauqzainuc
CUDA_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
vecadd-opensycl-cuda: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc -fuse-ld=lld --opensycl-targets="cuda:sm_80" -Wl,--rpath=$(LIBSTDCXX_DIR) -L$(LIBSTDCXX_DIR) -stdlib=libc++ -std=c++17 -I$(LLVM_DIR)/include -L$(LLVM_DIR)/lib -L$(CUDA_DIR)/lib64 -I$(OPENSYCL)/include -L$(OPENSYCL)/lib -L$(OPENSYCL)/lib/hipSYCL -L. -Wl,--rpath=$(LLVM_DIR)/lib/x86_64-unknown-linux-gnu -Wl,--rpath=$(LLVM_DIR)/lib -lopensycl-common -lhipSYCL-rt -lLLVM -lc++abi -o $@ $^
#	$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_80" -L$(OPENSYCL)/lib -L$(OPENSYCL)/lib/hipSYCL -L$(LLVM_DIR)/lib/x86_64-unknown-linux-gnu -stdlib=libc++ -std=c++17 -lc++ -lm -lhipSYCL-rt -lrt-backend-cuda -lopensycl-common -Wl,--rpath=$(LLVM_DIR)/lib/x86_64-unknown-linux-gnu -o $@ $^
#	$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_80" -L$(OPENSYCL)/lib -L$(OPENSYCL)/lib/hipSYCL -stdlib=libc++ -lhipSYCL-rt -lrt-backend-cuda -lopensycl-common -lopensycl-clang -o $@ $^
	#$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_86" -L$(OPENSYCL)/lib/hipSYCL -lrt-backend-cuda -L$(OPENSYCL)/lib -lhipSYCL-rt -lopensycl-common -stdlib=libc++ -lstdc++ -lm -o $@ $^
	#$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_86" --hipsycl-platform=cuda --hipsycl-gpu-arch=sm_62 -stdlib=libc++ -Wl,--rpath=$(LLVM_DIR)/lib/x87_64-unknown-linux-gnu -lstdc++ -lm -L$(OPENSYCL)/lib -L$(OPENSYCL)/lib/hipSYCL -lhipSYCL-rt -lrt-backend-cuda -L$(LLVM_DIR)/lib -o $@ $^

#broken
vecadd-opensycl-hip: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="hip:gfx908:sramecc+:xnack-" -stdlib=libc++  -o $@ $^

#broken
vecadd-computecpp: vecadd-sycl.cpp
	~/computecpp/bin/compute++ -I/home/9bj/computecpp/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/include -sycl-driver -sycl-target spir64 -o $@ $^


clean:
	rm -f vecadd vecadd-iris vecadd-m kernel.ptx kernel.openmp.so kernel.hip vecadd-iris-data-memory vecadd-opensycl-openmp vecadd-opensycl-openmp-discard-write vecadd-sycl vecadd-sycl-discard-write vecadd-sycl-dpc++ vecadd-sycl-dpc++-discard-write *.csv
