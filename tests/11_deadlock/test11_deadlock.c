#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define SIZE 128
#define LOOP 20

static void run(void* argp) {
  int id = *((int*) argp);
  printf("[%s:%d] id[%d]\n", __FILE__, __LINE__, id);
  iris_init(NULL, NULL, 1);
  void* host = malloc(SIZE);
  iris_mem mem;
  iris_mem_create(SIZE, &mem);

  for (int i = 0; i < LOOP; i++) {
    iris_task task;
    iris_task_create(&task);
    if (id & 1) {
      iris_task_d2h_full(task, mem, host);
    } else {
      iris_task_h2d_full(task, mem, host);
    }
    iris_task_submit(task, iris_gpu, NULL, 1);
    //iris_task_release(task);
    //printf("Error count:%d\n", iris_error_count());
  }

  free(host);
  //iris_finalize();
}

pthread_t t[256];

int main(int argc, char** argv) {
  int i;
  int id[128];
  int nthreads = argc > 1 ? atoi(argv[1]) : 10;
  printf("nthreads[%d]\n", nthreads);
  for (i = 0; i < 128; i++) id[i] = i;
  for (i = 0; i < nthreads; i++) {
    pthread_create(t + i, NULL, run, id + i);
  }
  for (i = 0; i < nthreads; i++) {
    pthread_join(t[i], NULL);
  }
  //printf("Joined all threads\n");
  printf("End errors:%d\n", iris_error_count());
  return iris_error_count();
}
