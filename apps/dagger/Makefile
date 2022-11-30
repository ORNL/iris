IRIS_INSTALL_ROOT?=$(HOME)/.local
CC=gcc
CXX=g++
NVCC?=$(CUDA_PATH)/bin/nvcc
FORTRAN=gfortran
CFLAGS=-g -std=c11 -I$(IRIR)/include -L$(IRIS)/lib
CXXFLAGS=-g  -std=c++11 -I$(IRIS)/include -L$(IRIS)/lib
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -liris -lpthread -ldl

all: dagger_runner dagger_test

dagger: dagger.cpp timer.h
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

dagger_dgemm: dagger_dgemm.cpp timer.h
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

#the final implementation?
dagger_runner: dagger_runner.cpp timer.h
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

dagger_test: dagger_test_concurrent_linear_chains.cpp timer.h
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

#dagger_coarse: dagger_coarse.cpp timer.h
#	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.spv: kernel.cl
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	#hipcc --genco --targets gfx906 -o $@ $^
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	#$(CC) -g -qsmp -fPIC -shared -I. -o $@ $^
	$(CC) -g -fopenmp -fPIC -shared -I. -o $@ $^

clean:
	rm -f dagger_runner dagger_test dagger dagger_dgemm dagger_coarse kernel.ptx kernel.spv kernel.hip kernel.openmp.so

clean-results:
	rm -f dagger_runner-*.csv timing_log_*.csv throughput-log-*.csv dagger_runner-*.html

save-results:
	mkdir -p dagger-results && mv dagger_runner-*.csv throughput-log-*.csv dagger-results
