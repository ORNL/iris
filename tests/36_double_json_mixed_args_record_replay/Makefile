TEST=test36_double_json_mixed_args_record_replay

all: $(TEST) kernel.ptx kernel.openmp.so kernel.hip

include ../Makefile.tests

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

