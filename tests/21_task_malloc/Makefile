TEST=test21_task_malloc

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

