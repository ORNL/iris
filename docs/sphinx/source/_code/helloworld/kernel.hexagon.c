#include <iris/iris_hexagon_imp.h>

AEEResult irishexagon_uppercase(char* b, int blen, char* a, int alen, IRIS_HEXAGON_KERNEL_ARGS) {
  int32 i = 0;
  IRIS_HEXAGON_KERNEL_BEGIN(i)
  if (a[i] >= 'a' && a[i] <= 'z') b[i] = a[i] + 'A' - 'a';
  else b[i] = a[i];
  IRIS_HEXAGON_KERNEL_END
  return AEE_SUCCESS;
}
