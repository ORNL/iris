TEST=test16_task_host

all: $(TEST) kernel.openmp.so kernel.ptx kernel.hip

include ../Makefile.tests

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

