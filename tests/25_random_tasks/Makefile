TEST=test25_random_tasks

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^
kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

