TEST=test07_policy_register

IRIS=$(HOME)/work/iris-rts

all: $(TEST) libPolicyGWS.so libPolicyGWSHook.so policy_last.so kernel.openmp.so

include ../Makefile.tests

policy_last.so: policy_last.cpp
	g++ -std=c++11 -fPIC -shared ${CXXFLAGS} -o $@ $^

libPolicyGWS.so: policy_gws.cpp
	g++ -std=c++11 -fPIC -shared ${CXXFLAGS} -o $@ $^

libPolicyGWSHook.so: policy_gws_hook.cpp
	g++ -std=c++11 -fPIC -shared ${CXXFLAGS} -o $@ $^

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.h
	gcc -g -fopenmp -fPIC -shared -I. ${CFLAGS} -o $@ kernel.cl.openmp.c
