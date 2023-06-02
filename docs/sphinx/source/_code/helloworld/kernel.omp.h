#include <iris/iris_openmp.h>

static void uppercase(char* b, char* a, IRIS_OPENMP_KERNEL_ARGS) {
  int i = 0;
#pragma omp parallel for shared(b, a) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  if (a[i] >= 'a' && a[i] <= 'z') b[i] = a[i] + 'A' - 'a';
  else b[i] = a[i];
  IRIS_OPENMP_KERNEL_END
}
