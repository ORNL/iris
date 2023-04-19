#include <iris/iris_openmp.h>

static void process(int* A, IRIS_OPENMP_KERNEL_ARGS) {
  size_t _id;
#pragma omp parallel for shared(A) private(_id)
  IRIS_OPENMP_KERNEL_BEGIN(_id)
  A[_id] = _id;
  IRIS_OPENMP_KERNEL_END
}

