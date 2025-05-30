CXX 					?= clang++
#CXX						:= xlC
NVCC					?= nvcc
HIPCC					?= hipcc
CFLAGS				:= -O3
CUDA_CFLAGS		?= -I/usr/local/cuda/include
CUDA_LDFLAGS	?= -lcuda
HIP_CFLAGS		?= -I/opt/rocm/hip/include -I/opt/rocm/hsa/include
HIP_LDFLAGS		?= -L/opt/rocm/hip/lib -lamdhip64
#HIPCC_FLAGS		+= --targets gfx906
OPENCL_CFLAGS			?= 
OPENCL_LDFLAGS		?= -lOpenCL
IRIS_LDFLAGS	:= -lbrisbane -lpthread -ldl

all: membench-iris-profiling

kernels: kernel.ptx kernel.hip

membench-iris-profiling: membench-iris-profiling.cpp
	$(CXX) $(CFLAGS) $(CXX_FLAGS) -g -o $@ $^ $(IRIS_LDFLAGS) $(LD_FLAGS)

kernel.ptx: kernel.cu
	$(NVCC) $(NVCC_FLAGS) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco $(HIPCC_FLAGS) -o $@ $^

clean:
	rm -f membench-iris-profiling kernel.ptx kernel.hip

clean-results:
	rm -f membench-*.csv

