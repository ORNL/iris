TEST=test07_policy_register

all: $(TEST) libPolicyGWS.so libPolicyGWSHook.so policy_last.so kernel.openmp.so kernel.hip kernel.ptx

include ../Makefile.tests

clean:
	rm -f $(TEST) kernel.ptx kernel.hip kernel.openmp.so policy_last.so libPolicyGWSHook.so libPolicyGWS.so

policy_last.so: policy_last.cpp
	g++ -std=c++11 -fPIC -shared ${CXXFLAGS} -o $@ $^

libPolicyGWS.so: policy_gws.cpp
	g++ -std=c++11 -fPIC -shared ${CXXFLAGS} -o $@ $^

libPolicyGWSHook.so: policy_gws_hook.cpp
	g++ -std=c++11 -fPIC -shared ${CXXFLAGS} -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -lgomp -Wl,-rpath=$(OPENMP_PATH) -fPIC -shared -I. $(CFLAGS) -o $@ $^
