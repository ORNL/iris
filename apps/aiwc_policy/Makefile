include ../makefile_defs.mk

all: libAIWCPolicy.so test_aiwc_policy

libAIWCPolicy.so: aiwc_policy.cpp aiwc_utils.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $^ -lcrypto $(LDFLAGS)

test_aiwc_policy: test_aiwc_policy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -lgomp -fPIC -shared -I. -o $@ $^

clean:
	rm -f libAIWCPolicy.so test_aiwc_policy kernel.ptx kernel.hip kernel.openmp.so

