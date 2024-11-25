IRIS_INSTALL_ROOT ?= $(HOME)/.iris
IRIS=$(IRIS_INSTALL_ROOT)

CHARMSYCL_INSTALL_ROOT ?= $(HOME)/.charm-sycl
CHARMSYCL=$(CHARMSYCL_INSTALL_ROOT)
OPENSYCL_INSTALL_ROOT ?= $(HOME)/.opensycl
OPENSYCL=$(OPENSYCL_INSTALL_ROOT)

CC ?= gcc
CXX ?= g++
FORTRAN ?= gfortran
NVCC ?= nvcc
HIPCC ?= hipcc
CHARMSYCL ?= $(HOME)/.charm-sycl
CHARMSYCL_LDFLAGS ?= -L$(CHARMSYCL)/lib -L$(CHARMSYCL)/lib64 -lcharm -lpthread -ldl
DPCPP ?= $(HOME)/dpc++-workspace

CFLAGS=-I$(IRIS)/include/ -O3 -std=c99
CXXFLAGS=-I$(IRIS)/include/ -O3
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl

