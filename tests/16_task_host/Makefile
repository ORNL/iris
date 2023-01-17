TEST=test16_task_host

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.h
	gcc -g -fopenmp -fPIC -shared -I. -o $@ kernel.cl.openmp.c
