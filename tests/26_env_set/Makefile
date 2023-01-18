CPP=1

TEST=test26_env_set

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel-negative.ptx: kernel-negative.cu
	nvcc -ptx $^
