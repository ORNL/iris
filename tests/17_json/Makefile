TEST=test17_json

all: $(TEST) kernel.openmp.so kernel.hip kernel.ptx

include ../Makefile.tests

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

clean:
	rm -f kernel.ptx kernel.hip kernel.openmp.so test17_json
