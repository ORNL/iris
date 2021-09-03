#CC=xlc
#CXX=xlC
CC=gcc
CXX=g++

all: sgemm

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	#hipcc --genco --targets gfx906 -o $@ $^
	hipcc --genco -o $@ $^


kernel.openmp.so: kernel.cl.openmp.c
	#$(CC) -O3 -qsmp -fPIC -shared -I. -o $@ $^
	$(CC) -std=c99 -fopenmp -fPIC -shared -I. -o $@ $^

sgemm: sgemm.c
	$(CC) -O3 -std=c99 -o $@ $^ -lbrisbane -lpthread -ldl

sgemm-m: sgemm-m.c
	$(CC) -O3 -std=c99 -o $@ $^ -lbrisbane -lpthread -ldl

clean:
	rm -f sgemm sgemm-brisbane libbrisbane-poly.so *.dot *.html
