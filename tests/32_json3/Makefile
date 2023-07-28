CPP=1

TEST=test32_json3
all: $(TEST) kernel.openmp.so kernel.ptx kernel.hip

include ../Makefile.tests

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

kernel.openmp.so: kernel.openmp.c
		$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^

kernel.spv: kernel.cl
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

clean:
	rm -f kernel.ptx kernel.spv kernel.hip kernel.openmp.so $(TEST)
