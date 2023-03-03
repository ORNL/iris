#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

double t0, t1;
double i0, i1;
double f0, f1;
double c0, c1;

int SIZE;
int LOOP;
size_t nbytes;
float A;
int REPEATS;

char* LOGFILE = NULL;
FILE* LF_HANDLE;
char LOG_BUFFER[32];

CUdevice dev;
CUcontext ctx;
CUdeviceptr dZ, dX, dY;
CUmodule mod;
CUfunction func = NULL;
CUresult err;

void Init() {
  int count;
  char name[256];
  err = cuInit(0);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuDeviceGetCount(&count);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuDeviceGet(&dev, 0);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuDeviceGetName(name, 256, dev);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, dev);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuMemAlloc(&dZ, nbytes);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuMemAlloc(&dX, nbytes);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuMemAlloc(&dY, nbytes);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuModuleLoad(&mod, "kernel.ptx");
  if (err != CUDA_SUCCESS){ printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err); exit(1);};
}

void Compute(int loop) {
  if (!func){
    err = cuModuleGetFunction(&func, mod, "saxpy");
    if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  }
  void* params[4] = { &dZ, &A, &dX, &dY };
  for (int i = 0; i < loop; i++)  {
    err = cuLaunchKernel(func, 1, 1, 1, SIZE, 1, 1, 0, 0, params, NULL);
  }
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuCtxSynchronize();
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
}

void Finalize() {
  err = cuMemFree(dZ);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuMemFree(dX);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuMemFree(dY);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuModuleUnload(mod);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = cuCtxDestroy(ctx);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  SIZE = atoi(argv[1]);
  LOOP = atoi(argv[2]);
  nbytes = SIZE * sizeof(float);
  REPEATS = 1;
  if (argc == 5) {
    REPEATS = atoi(argv[3]);
    LOGFILE = argv[4];
  }

  printf("[%s:%d] SIZE[%d][%luB] LOOP[%d] REPEATS[%d]\n", __FILE__, __LINE__, SIZE, nbytes, LOOP, REPEATS);
  t0 = now();

  i0 = now();
  Init();
  i1 = now();

  for (int i = 0; i < REPEATS; ++ i) {
    c0 = now();
    Compute(LOOP);
    c1 = now();

    printf("latency [%lf] ms\n", ((c1 - c0) / LOOP) * 1.e+6);
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

  f0 = now();
  Finalize();
  f1 = now();

  t1 = now();

  printf("secs: T[%lf] I[%lf] C[%lf] F[%lf]\n", t1 - t0, i1 - i0, c1 - c0, f1 - f0);

  return 0;
}

