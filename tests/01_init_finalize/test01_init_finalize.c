#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);
  iris_finalize();
  printf("Error count:%d\n", iris_error_count());
  return iris_error_count();
}
