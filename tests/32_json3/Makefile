CPP=1

TEST=test32_json3

include ../Makefile.tests

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.spv: kernel.cl
	clang -cc1 -finclude-default-header -triple spir $^ -O0 -emit-llvm-bc -o kernel.bc
	llvm-spirv kernel.bc -o $@

kernel.hip: kernel.hip.cpp
	#hipcc --genco --targets gfx906 -o $@ $^
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -fopenmp -fPIC -shared -I. -o $@ $^

clean:
	rm -f kernel.ptx kernel.spv kernel.hip kernel.openmp.so $(TEST)
