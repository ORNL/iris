#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

static int run(void* argp) {
  iris_init(NULL, NULL, 1);
  iris_finalize();
  return iris_error_count();
}

pthread_t t[256];

int main(int argc, char** argv) {
  //setenv("IRIS_ARCHS", "opencl", 1);
  int i;
  int nthreads = argc > 1 ? atoi(argv[1]) : 10;
  printf("nthreads[%d]\n", nthreads);
  for (i = 0; i < nthreads; i++) {
    pthread_create(t + i, NULL, run, NULL);
  }
  int returnval = 0;
  for (i = 0; i < nthreads; i++) {
    void* rv;
    pthread_join(t[i], &rv);
    returnval += (int)(off_t)rv;
  }
  return returnval;
}
