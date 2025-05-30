#include "timer.h"
#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

double t0, t1;
double i0, i1;
double f0, f1;
double c0, c1;

int LOOP;

int err;

void Init() {
  err = brisbane_init(NULL, NULL, 1);
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
}

void Compute() {
  for (int i = 0; i < LOOP; i++) {
    brisbane_task task;
    err = brisbane_task_create(&task);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
    err = brisbane_task_submit(task, 0, NULL, 0);
    if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
  }
  err = brisbane_synchronize();
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
}

void Finalize() {
  err = brisbane_finalize();
  if (err != BRISBANE_OK) printf("[%s:%d]\n", __FILE__, __LINE__);
}

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  LOOP = atoi(argv[1]);
  printf("[%s:%d] LOOP[%d]\n", __FILE__, __LINE__, LOOP);

  t0 = now();

  i0 = now();
  Init();
  i1 = now();

  c0 = now();
  Compute();
  c1 = now();

  f0 = now();
  Finalize();
  f1 = now();

  t1 = now();

  printf("latency [%lf] us\n", ((c1 - c0) / LOOP) * 1.e+6);
  printf("secs: T[%lf] I[%lf] C[%lf] F[%lf]\n", t1 - t0, i1 - i0, c1 - c0, f1 - f0);

  return 0;
}

