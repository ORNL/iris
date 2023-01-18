#include <brisbane/brisbane_openmp.h>

static void process(int* A, BRISBANE_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(A) private(_id)
  BRISBANE_OPENMP_KERNEL_BEGIN
  A[_id] *= 100;
  BRISBANE_OPENMP_KERNEL_END
}

