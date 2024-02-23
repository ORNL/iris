TEST=test23_multigraph

all: $(TEST) kernel.ptx kernel.openmp.so kernel.hip

include ../Makefile.tests

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

run: $(TEST) kernel.ptx
	./$(TEST) 4096 6 5
