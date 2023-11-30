CPP=1

TEST=test26_env_set
all: $(TEST) kernel.ptx kernel-negative.ptx

include ../Makefile.tests

kernel-negative.ptx: kernel-negative.cu
	$(NVCC) -ptx $^
