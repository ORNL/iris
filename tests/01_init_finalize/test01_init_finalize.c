#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);
  iris_finalize();
  return iris_error_count();
}
