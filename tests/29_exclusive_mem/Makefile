TEST=test29_exclusive_mem

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^
