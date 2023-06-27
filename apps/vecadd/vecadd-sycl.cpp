#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B, *C;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%lu]\n", SIZE);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));
  C = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i;
    C[i] = 0;
  }
  {
    sycl::buffer<int, 1> mem_A(A, {SIZE});
    sycl::buffer<int, 1> mem_B(B, {SIZE});
    sycl::buffer<int, 1> mem_C(C, {SIZE});
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
        sycl::accessor<int, 1, sycl::access_mode::read>  d_a(mem_A, h);
        sycl::accessor<int, 1, sycl::access_mode::read>  d_b(mem_B, h);
        sycl::accessor<int, 1, sycl::access_mode::write> d_c(mem_C, h);
        h.parallel_for(sycl::range{SIZE}, [=](sycl::id<1> i) {
            d_c[i] = d_a[i] + d_b[i];
          });
        });
    q.wait();
  }
  for (int i = 0; i < SIZE; i++) {
    printf("C[%d] = %d\n", i, C[i]);
    if (C[i] != (A[i] + B[i])) ERROR++;
  }
  return ERROR;
}

