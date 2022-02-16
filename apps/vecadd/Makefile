include ../makefile_defs.mk

all: vecadd vecadd-iris vecadd-m

vecadd: vecadd.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-iris: vecadd-iris.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-m: vecadd-m.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

clean:
	rm -f vecadd vecadd-iris vecadd-m *.dot *.html
