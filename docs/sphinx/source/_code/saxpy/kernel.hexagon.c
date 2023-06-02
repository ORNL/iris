#include <iris/iris_hexagon_imp.h>

AEEResult irishexagon_saxpy(float* S, int Slen, float A, float* X, int Xlen, float* Y, int Ylen, IRIS_HEXAGON_KERNEL_ARGS) {
  int32 i = 0;
  IRIS_HEXAGON_KERNEL_BEGIN(i)
  S[i] = A * X[i] + Y[i];
  IRIS_HEXAGON_KERNEL_END
  return AEE_SUCCESS;
}
