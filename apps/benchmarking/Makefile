NVCC					?= $(CUDA_PATH)/bin/nvcc
HIPCC					?= $(ROCM_PATH)/bin/hipcc
CFLAGS				:= -O3
CUDA_CFLAGS		?= -I$(CUDA_PATH)/include
CUDA_LDFLAGS	?= -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/lib64 -lcuda -lcudart
#HIP_CFLAGS		?= -I$(ROCM_PATH)/include -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hsa/include
#HIP_LDFLAGS		?= -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/hip/lib -lamdhip64 -D__HIP_PLATFORM_AMD__
-HIP_CFLAGS		?= -I$(ROCM_PATH)/include -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hsa/include
-HIP_LDFLAGS	?= -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/hip/lib -lamdhip64
##HIPCC_FLAGS		+= --targets gfx906
OPENCL_CFLAGS			?= -I$(OPENCL_PATH)/include
OPENCL_LDFLAGS		?= -L$(OPENCL_PATH)/lib -L$(OPENCL_PATH)/lib64 -lOpenCL
IRIS					:=$(HOME)/.iris
IRIS_LDFLAGS	:=-L$(IRIS)/lib -L$(IRIS)/lib64 -liris -lpthread -ldl
CHARMSYCL 		:=$(HOME)/.charm-sycl
CHARMSYCL_LDFLAGS	:=-L$(CHARMSYCL)/lib -L$(CHARMSYCL)/lib64 -lcharm -lpthread -ldl

kernels: kernel.ptx kernel.hip kernel.openmp.so

#TODO: what do to with empty-task?
#emptytask: emptytask-iris
#
#emptytask-iris: emptytask-iris.cpp
#	$(CXX) $(CXX_FLAGS) -o $@ $^ $(IRIS_LDFLAGS)

#TODO: set up OpenMP measurement
#task-latency-openmp: task-latency-codes/kernellaunch-openmp.cpp

task-latency-cuda: task-latency-codes/kernellaunch-cuda-profiling.cpp
	$(CXX) $(CUDA_CFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

task-latency-hip: task-latency-codes/kernellaunch-hip-profiling.cpp
	$(CXX) $(HIP_CFLAGS) -o $@ $^ $(HIP_LDFLAGS)

task-latency-opencl: task-latency-codes/kernellaunch-opencl-profiling.cpp
	$(CXX) $(OPENCL_CFLAGS) -o $@ $^ $(OPENCL_LDFLAGS)

task-latency-openmp: task-latency-codes/kernellaunch-openmp-profiling.cpp
	$(CXX) -lgomp -Wl,-rpath=$(OPENMP_PATH) -I$(IRIS)/include $(INCLUDE) $(EXC_INCLUDE) -o $@ $^ $(IRIS_LDFLAGS)

task-latency-iris: task-latency-codes/kernellaunch-iris.cpp
	$(CXX) $(CFLAGS) $(CXX_FLAGS) -I$(IRIS)/include  -o $@ $^ $(IRIS_LDFLAGS) $(LD_FLAGS)

compute-performance-iris: compute-performance-codes/flops-dgemm-iris-profiling.cpp
	$(CXX) -I$(IRIS)/include -o $@ $^ $(IRIS_LDFLAGS)

memory-performance-iris: memory-performance-codes/membench-iris-profiling.cpp
	$(CXX) -I$(IRIS)/include -o $@ $^ $(IRIS_LDFLAGS)

memory-performance-sycl: memory-performance-codes/membench-charmsycl-profiling.cpp
	$(CHARMSYCL)/bin/cscc --targets=cpu-c,nvidia-cuda -o $@ $^

kernel.ptx: kernel.cu
	$(NVCC) $(NVCC_FLAGS) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco $(HIPCC_FLAGS) -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -I$(IRIS)/include -o $@ $^

clean:
	rm -f task-latency-cuda task-latency-hip task-latency-opencl task-latency-openmp compute-performance-iris memory-performance-iris kernel.ptx kernel.hip

clean-results:
	rm -f kernellaunch-*.csv

