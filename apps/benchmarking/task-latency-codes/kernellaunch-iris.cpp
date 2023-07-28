#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <brisbane/brisbane.h>

double t0, t1;
double i0, i1;
double f0, f1;
double c0, c1;
double r0, r1;

int SIZE;
int LOOP;
int REPEATS;
size_t nbytes;

char* LOGFILE = NULL;
FILE* LF_HANDLE;
char LOG_BUFFER[32];

float A;

brisbane_mem dZ, dX, dY;
int err;

void Init() {
  err = brisbane_init(NULL, NULL, 1);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = brisbane_mem_create(nbytes, &dZ);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = brisbane_mem_create(nbytes, &dX);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = brisbane_mem_create(nbytes, &dY);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
}

void Compute(int loop) {
  size_t gws = (size_t) SIZE;
  void* params[4] = { dZ, &A, dX, dY };
  int pinfo[4] = { brisbane_w, sizeof(A), brisbane_r, brisbane_r };
  for (int i = 0; i < loop - 1; i++) {
    brisbane_task task;
    err = brisbane_task_create(&task);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    err = brisbane_task_kernel(task, "saxpy", 1, NULL, &gws, &gws, 4, params, pinfo);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    err = brisbane_task_submit(task, 0, NULL, 0);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    //err = brisbane_task_release(task);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  }
  for (int i = loop - 1; i < loop; i++) {
    brisbane_task task;
    err = brisbane_task_create(&task);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    err = brisbane_task_kernel(task, "saxpy", 1, NULL, &gws, &gws, 4, params, pinfo);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    err = brisbane_task_submit(task, 0, NULL, 1);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    //err = brisbane_task_release(task);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  }
  //err = brisbane_synchronize();
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
}

void Finalize() {
  err = brisbane_mem_release(dZ);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = brisbane_mem_release(dX);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = brisbane_mem_release(dY);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = brisbane_finalize();
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  SIZE = atoi(argv[1]);
  LOOP = atoi(argv[2]);
  REPEATS = 1;
  if (argc == 5) {
    REPEATS = atoi(argv[3]);
    LOGFILE = argv[4];
  }
  nbytes = SIZE * sizeof(float);
  printf("[%s:%d] SIZE[%d][%luB] LOOP[%d] REPEATS[%d]\n", __FILE__, __LINE__, SIZE, nbytes, LOOP, REPEATS);

  for (int i = 0; i < REPEATS; ++ i) {
    t0 = now();

    i0 = now();
    Init();
    i1 = now();

    Compute(1);

    c0 = now();
    Compute(LOOP);
    c1 = now();

    f0 = now();
    Finalize();
    f1 = now();

    t1 = now();

    printf("latency [%lf] ms\n", ((c1 - c0) / LOOP) * 1.e+6);
    printf("secs: T[%lf] I[%lf] C[%lf] F[%lf]\n", t1 - t0, i1 - i0, c1 - c0, f1 - f0);

    if (LOGFILE != NULL) {
      LF_HANDLE = fopen(LOGFILE, "a");
      assert(LF_HANDLE != NULL);
      if(i == 0){
        sprintf(LOG_BUFFER, "%g", (c1-c0)*1.e+6);
      }
      else {
        sprintf(LOG_BUFFER, ",%g", (c1-c0)*1.e+6);
      }
      fputs(LOG_BUFFER,LF_HANDLE);
      fclose(LF_HANDLE);
    }
  }
  if (LOGFILE != NULL) {
      LF_HANDLE = fopen(LOGFILE, "a");
      assert(LF_HANDLE != NULL);
      fputs("\n", LF_HANDLE);
      fclose(LF_HANDLE);
  }

  return 0;
}

