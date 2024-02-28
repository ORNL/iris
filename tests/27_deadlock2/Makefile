CPP=1

TEST=test27_deadlock2

all: $(TEST) kernel.openmp.so kernel.hip kernel.ptx

include ../Makefile.tests

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^
