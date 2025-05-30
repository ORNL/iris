TEST=test34_set_mem
CPP=1

all: $(TEST) kernel.ptx kernel.openmp.so kernel.hip

include ../Makefile.tests

vecadd: vecadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-iris: vecadd-iris.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.openmp.so: kernel.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

ifeq ($(NVCC_TEST),)
kernel.nvopenmp.so: kernel.openmp.c
	@echo "No NVCC compiler found"
else
kernel.nvopenmp.so: kernel.openmp.c
	$(NVCC) -g -shared -I. ${INCLUDE} ${EXT_INCLUDE} --compiler-options -fPIC -o $@ $^
endif

clean:
	rm -f vecadd vecadd-iris vecadd-m kernel.ptx kernel.nvopenmp.so kernel.openmp.so kernel.hip test34_set_mem
