#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  char vendor[64];
  char name[64];
  int type;
  int nplatforms = 0;
  int ndevs = 0;
  iris_platform_count(&nplatforms);
  for (int i = 0; i < nplatforms; i++) {
    size_t size;
    iris_platform_info(i, iris_name, name, &size);
    printf("platform[%d] name[%s]\n", i, name);
  }

  iris_device_count(&ndevs);
  for (int i = 0; i < ndevs; i++) {
    size_t size;
    iris_device_info(i, iris_vendor, vendor, &size);
    iris_device_info(i, iris_name, name, &size);
    iris_device_info(i, iris_type, &type, &size);
    printf("dev[%d] vendor[%s] name[%s] type[0x%x]\n", i, vendor, name, type);
  }
  iris_finalize();

  return iris_error_count();
}
