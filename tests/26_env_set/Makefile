CPP=1

TEST=test26_env_set
all: $(TEST) kernel.ptx kernel-negative.ptx

include ../Makefile.tests

ifeq ($(NVCC_TEST),)
kernel-negative.ptx: kernel-negative.cu
	@echo "No NVCC compiler found"
else
kernel-negative.ptx: kernel-negative.cu
	$(NVCC) -ptx $^
endif
