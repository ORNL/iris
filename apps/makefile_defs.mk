IRIS_INSTALL_ROOT ?= $(HOME)/.local
IRIS=$(IRIS_INSTALL_ROOT)

CC ?= gcc
CXX ?= g++
FORTRAN ?= gfortran
NVCC ?= nvcc
HIPCC ?= hipcc

CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
CXXFLAGS=-I$(IRIS)/include/ -O3
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl

