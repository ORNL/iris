#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "iris/iris_openmp.h"
#include "iris/gettime.h"
#include "iris_app_cpu_dsp_interface.h"

int saxpy(int32_t* Z, const int32_t* X, const int32_t* Y, int32_t SIZE, int32_t A, IRIS_OPENMP_KERNEL_ARGS)
//int saxpy(int32_t* Z, int32_t A, const int32_t* X, const int32_t* Y, int32_t SIZE, int32_t *dspUsec, int32_t *dspCyc, IRIS_OPENMP_KERNEL_ARGS) 
{
  size_t i;
  //printf("Kernel Launch parameters: X:%p Y:%p Z:%p A:%d xSize:%d dspUsec:%p dspCyc:%p\n", X, Y, Z, A, SIZE, dspUsec, dspCyc);
  //IRIS_OPENMP_KERNEL_BEGIN(i)
  printf("A:%d SIZE:%d\n", A, SIZE);
  #pragma omp parallel for shared(Z, A, X, Y) private(i)
  for(i=0; i<SIZE; i++)
      Z[i] = A * X[i] + Y[i];
  //IRIS_OPENMP_KERNEL_END
  //printf("Kernel Launch parameters: X:%p Y:%p Z:%p A:%d xSize:%d dspUsec:%p dspCyc:%p\n", X, Y, Z, A, SIZE, dspUsec, dspCyc);
  return 0;
}
