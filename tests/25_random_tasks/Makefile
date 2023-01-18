TEST=test25_random_tasks

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^
