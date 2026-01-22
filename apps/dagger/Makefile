include ../makefile_defs.mk
CXX=g++
CC=gcc

all: dagger_runner dagger_test

dagger: dagger.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

dagger_dgemm: dagger_dgemm.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

dagger_runner: dagger_runner.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

dagger_test: dagger_test_concurrent_linear_chains.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -allow-unsupported-compiler -ptx $^

kernel.spv: kernel.cl
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

#kernel.openmp.so: kernel.openmp.c
#	$(CC) $(CFLAGS) -O3 -lgomp -fPIC -shared -I. -o $@ $^

libPolicyGNN.so: PolicyGNN.cpp
	$(CXX) $(CXXFLAGS) -std=c++11 -fPIC -shared -o $@ $^ -DRECORD_GNN_OVERHEAD

clean:
	rm -f dagger_runner dagger_test dagger dagger_dgemm dagger_coarse kernel.ptx kernel.spv kernel.hip kernel.openmp.so

clean-results:
	rm -f dagger_runner-*.csv timing_log_*.csv throughput-log-*.csv dagger_runner-*.html

save-results:
	mkdir -p dagger-results && mv dagger_runner-*.csv throughput-log-*.csv dagger-results
