TEST=test07_policy_register

IRIS=$(HOME)/work/iris-rts

all: $(TEST)

include ../Makefile.tests

policy_last.so: policy_last.cpp
	g++ -std=c++11 -fPIC -shared -o $@ $^

libPolicyGWS.so: policy_gws.cpp
	g++ -std=c++11 -fPIC -shared -o $@ $^

libPolicyGWSHook.so: policy_gws_hook.cpp
	g++ -std=c++11 -fPIC -shared -o $@ $^

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.openmp.so: kernel.cl.openmp.h
	gcc -g -fopenmp -fPIC -shared -I. -o $@ kernel.cl.openmp.c
