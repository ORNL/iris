IRIS=$(HOME)/.local
CC=gcc
CXX=g++
FORTRAN=gfortran
CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
CXXFLAGS=-I$(IRIS)/include/ -O3
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl

all: twotasks kernel.openmp.so

twotasks: twotasks.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

clean:
	rm -f twotasks kernel.openmp.so
