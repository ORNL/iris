#CXX 					:= clang++
#CXX						:= xlC
NVCC					?= $(CUDA_PATH)/bin/nvcc
HIPCC					?= $(ROCM_PATH)/bin/hipcc
CFLAGS				:= -O3
CUDA_CFLAGS		?= -I$(CUDA_PATH)/include
CUDA_LDFLAGS	?= -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/lib64 -lcuda -lcudart
HIP_CFLAGS		?= -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hsa/include
HIP_LDFLAGS		?= -L$(ROCM_PATH)/hip/lib -lamdhip64
#HIPCC_FLAGS		+= --targets gfx906
OPENCL_CFLAGS			?= -I$(OPENCL_PATH)/include
OPENCL_LDFLAGS		?= -L$(OPENCL_PATH)/lib -L$(OPENCL_PATH)/lib64 -lOpenCL
IRIS_LDFLAGS	:=-L$(IRIS)/lib -L$(IRIS)/lib64 -liris -lpthread -ldl

all: kernellaunch

kernels: kernel.ptx kernel.hip

emptytask: emptytask-iris

emptytask-iris: emptytask-iris.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(IRIS_LDFLAGS)

kernellaunch: kernellaunch-cuda kernellaunch-hip kernellaunch-opencl kernellaunch-iris

kernellaunch-prof: kernellaunch-cuda-profiling kernellaunch-iris-profiling

kernellaunch-cuda: kernellaunch-cuda.cpp
	$(CXX) $(CUDA_CFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

kernellaunch-cuda-profiling: kernellaunch-cuda-profiling.cpp
	$(CXX) $(CUDA_CFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

kernellaunch-hip: kernellaunch-hip.cpp
	$(HIPCC) $(HIP_CFLAGS) -o $@ $^ $(HIP_LDFLAGS)

kernellaunch-hip-profiling: kernellaunch-hip-profiling.cpp
	$(CXX) $(HIP_CFLAGS) -o $@ $^ $(HIP_LDFLAGS)

kernellaunch-opencl: kernellaunch-opencl.cpp
	$(CXX) $(OPENCL_CFLAGS) -o $@ $^ $(OPENCL_LDFLAGS)

kernellaunch-opencl-profiling: kernellaunch-opencl-profiling.cpp
	$(CXX) $(OPENCL_CFLAGS) -o $@ $^ $(OPENCL_LDFLAGS)

kernellaunch-iris: kernellaunch-iris.cpp
	$(CXX) $(CFLAGS) $(CXX_FLAGS) -I$(IRIS)/include -o $@ $^ $(IRIS_LDFLAGS) $(LD_FLAGS)

kernellaunch-iris-profiling: kernellaunch-iris-profiling.cpp
	$(CXX) $(CFLAGS) $(CXX_FLAGS) -I$(IRIS)/include -DNDEBUG -o $@ $^ $(IRIS_LDFLAGS) $(LD_FLAGS)

compute-performance-iris-profiling: flops-dgemm-iris-profiling.cpp
	$(CXX) -I$(IRIS)/include -o $@ $^ $(IRIS_LDFLAGS)

memory-performance-iris-profiling: membench-iris-profiling.cpp
	$(CXX) -I$(IRIS)/include -o $@ $^ $(IRIS_LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) $(NVCC_FLAGS) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco $(HIPCC_FLAGS) -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -g -std=c99 -fopenmp -fPIC -shared -I. -o $@ $^

clean:
	rm -f emptytask-iris kernellaunch-cuda kernellaunch-hip kernellaunch-opencl kernellaunch-iris kernellaunch-cuda-profiling kernellaunch-hip-profiling kernellaunch-opencl-profiling kernellaunch-iris-profiling compute-performance-iris-profiling kernel.ptx kernel.hip

clean-results:
	rm -f kernellaunch-*.csv

