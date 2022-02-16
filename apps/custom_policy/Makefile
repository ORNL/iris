include ../makefile_defs.mk

all: custom_policy kernel.openmp.so libPolicyGWS.so

custom_policy: custom_policy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -I. -o $@ $^

libPolicyGWS.so: PolicyGWS.cpp
	$(CXX) $(CXXFLAGS) -std=c++11 -fPIC -shared -o $@ $^

clean:
	rm -f custom_policy kernel.openmp.so libPolicyGWS.so
