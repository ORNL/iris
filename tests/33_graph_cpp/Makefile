TEST=test33_graph_cpp
CPP=1
all: $(TEST) kernel.openmp.so kernel.ptx kernel.hip

include ../Makefile.tests

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^
