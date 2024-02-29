TEST=test19_replay

all: $(TEST) kernel.openmp.so kernel.hip kernel.ptx

include ../Makefile.tests

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

cleanjson:
	rm -f *.json
