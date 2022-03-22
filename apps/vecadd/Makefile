include ../makefile_defs.mk

all: vecadd vecadd-iris vecadd-m

vecadd: vecadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-iris: vecadd-iris.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

vecadd-m: vecadd-m.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $^

clean:
	rm -f vecadd vecadd-iris vecadd-m *.dot *.html
