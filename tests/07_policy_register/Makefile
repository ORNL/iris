TEST=test07_policy_register

all: $(TEST) libPolicyGWS.so libPolicyGWSHook.so policy_last.so kernel.openmp.so kernel.hip kernel.ptx

include ../Makefile.tests

clean:
	rm -f $(TEST) kernel.ptx kernel.hip kernel.openmp.so policy_last.so libPolicyGWSHook.so libPolicyGWS.so

policy_last.so: policy_last.cpp
	$(CXX) -shared ${CXXFLAGS} -o $@ $^

libPolicyGWS.so: policy_gws.cpp
	$(CXX) -shared ${CXXFLAGS} -o $@ $^ -liris ${LDFLAGS}

libPolicyGWSHook.so: policy_gws_hook.cpp
	$(CXX) -shared ${CXXFLAGS} -o $@ $^ -liris ${LDFLAGS}

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g $(IRIS_OPENMP_PATH) -shared -I. $(CFLAGS) -o $@ $^
