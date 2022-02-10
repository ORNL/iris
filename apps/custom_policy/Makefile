IRIS_INSTALL_ROOT ?= $(HOME)/.local
IRIS=$(IRIS_INSTALL_ROOT)
CC=gcc
CXX=g++
FORTRAN=gfortran
CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
CXXFLAGS=-I$(IRIS)/include/ -O3
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl

all: custom_policy kernel.openmp.so libPolicyGWS.so

custom_policy: custom_policy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

libPolicyGWS.so: PolicyGWS.cpp
	$(CXX) -std=c++11 -fPIC -shared -o $@ $^

clean:
	rm -f custom_policy kernel.openmp.so libPolicyGWS.so
