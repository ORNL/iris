#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

static void* run(void* argp) {
  iris_init(NULL, NULL, 1);
  iris_finalize();
}

pthread_t t[256];

int main(int argc, char** argv) {
  int i;
  int nthreads = argc > 1 ? atoi(argv[1]) : 10;
  printf("nthreads[%d]\n", nthreads);
  for (i = 0; i < nthreads; i++) {
    pthread_create(t + i, NULL, run, NULL);
  }
  for (i = 0; i < nthreads; i++) {
    pthread_join(t[i], NULL);
  }
 
  return 0;
}
