include ../makefile_defs.mk

all: vecadd vecadd-iris kernel.ptx kernel.openmp.so kernel.hip

###############################################################################
#                                 IRIS  Kernels                               #
###############################################################################

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

###############################################################################
#                               Vecadd Benchmark                              #
###############################################################################
# OpenARC
vecadd: vecadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
# IRIS
vecadd-iris: vecadd-iris.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
# Charm-SYCL
vecadd-sycl: vecadd-sycl.cpp
	$(CHARMSYCL)/bin/cscc --targets=all $(CXXFLAGS) -o $@ $^
# DPC++ (CUDA)
vecadd-sycl-dpc++: vecadd-sycl.cpp
	$(DPCPP)/llvm/build/bin/clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda $(CXXFLAGS) -o $@ $^
# OpenSYCL (CUDA)
# pending... TODO!
# OpenSYCL (OpenMP)
vecadd-opensycl-openmp: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -pthread -lgomp -O3 -o $@ $^

#broken
#LLVM_DIR=/home/9bj/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-12.1.0/llvm-13.0.1-2a5yipjwa6bco5vymttszyqauqzainuc
CUDA_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
vecadd-opensycl-cuda: vecadd-sycl.cpp
	$(OPENSYCL)/bin/syclcc-clang --opensycl-targets="cuda:sm_80" -O3 -o $@ $^

#	$(OPENSYCL)/bin/syclcc --opensycl-targets="cuda:sm_80" -Wl,--rpath=$(LIBSTDCXX_DIR) -L$(LIBSTDCXX_DIR) -stdlib=libc++ -std=c++17 -I$(LLVM_DIR)/include -L$(LLVM_DIR)/lib -L$(CUDA_DIR)/lib64 -I$(OPENSYCL)/include -L$(OPENSYCL)/lib -L$(OPENSYCL)/lib/hipSYCL -Wl,--rpath=$(LLVM_DIR)/lib/x86_64-unknown-linux-gnu -Wl,--rpath=$(LLVM_DIR)/lib  -lc++ -lc++abi -lLLVM -lopensycl-common -lhipSYCL-rt -o $@ $^
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


###############################################################################
#                            Optimization Benchmark                           #
#         data-memory optimization should remove 1 extra memory transfer      #
# sycl discard_write should also avoid 1 extra iris_task_h2d() being called.  #
# this test examines if we can get this gain for free in Charm-SYCL with IRIS!#
###############################################################################
# IRIS 
vecadd-iris-data-memory: vecadd-iris-data-memory.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
# Charm-SYCL
vecadd-sycl-discard-write: vecadd-sycl-discard-write.cpp
	$(CHARMSYCL)/bin/cscc --targets=all $(CXXFLAGS) -o $@ $^
# DPC++ (CUDA)
vecadd-sycl-dpc++-discard-write: vecadd-sycl-discard-write.cpp
	$(DPCPP)/llvm/build/bin/clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda $(CXXFLAGS) -o $@ $^
# OpenSYCL (CUDA)
# pending... TODO!
# OpenSYCL (OpenMP)
vecadd-opensycl-openmp-discard-write: vecadd-sycl-discard-write.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -pthread -lgomp $(CXXFLAGS) -o $@ $^

###############################################################################
#                                Empty Benchmark                              #
###############################################################################
# IRIS
empty-iris: empty-iris.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
# Charm-SYCL
empty-sycl: empty-sycl.cpp
	$(CHARMSYCL)/bin/cscc --targets=all $(CXXFLAGS) -o $@ $^
# DPC++ (CUDA)
empty-sycl-dpc++: empty-sycl.cpp
	$(DPCPP)/llvm/build/bin/clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda $(CXXFLAGS) -o $@ $^
# OpenSYCL (CUDA)
# pending... TODO!
# OpenSYCL (OpenMP)
empty-opensycl-openmp: empty-sycl.cpp
	$(OPENSYCL)/bin/syclcc --opensycl-targets="omp" -pthread -lgomp -O3 -o $@ $^

clean:
	rm -f vecadd vecadd-iris vecadd-m kernel.ptx kernel.openmp.so kernel.hip \
		vecadd-sycl vecadd-opensycl-openmp vecadd-sycl-dpc++ \
		vecadd-iris-data-memory vecadd-sycl-discard-write vecadd-opensycl-openmp-discard-write vecadd-sycl-dpc++-discard-write\
		empty-iris empty-sycl empty-sycl-dpc++ empty-opensycl-openmp \
		*.csv
