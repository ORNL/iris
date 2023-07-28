CPP=1

TEST=test26_env_set
all: $(TEST) kernel.ptx kernel-negative.ptx

include ../Makefile.tests

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel-negative.ptx: kernel-negative.cu
	$(NVCC) -ptx $^
