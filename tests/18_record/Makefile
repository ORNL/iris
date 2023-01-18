TEST=test18_record

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	gcc -g -fopenmp -fPIC -shared -I. -o $@ $^

test18_replay:test18_replay.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDINC) $(LDFLAGS)

cleanjson:
	rm -f *.json
