include ../makefile_defs.mk

all: kernel.openmp.so

kernel.openmp.so: post_processing.cpp
	$(CXX) $(CXXFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

clean:
	rm -f kernel.openmp.so
