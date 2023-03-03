#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>

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

hipDevice_t dev;
hipCtx_t ctx;
void *dZ, *dX, *dY;
hipModule_t mod;
hipFunction_t func = NULL;
hipError_t err;
hipEvent_t start, stop;

void Init() {
  int count;
  char name[256];
  err = hipInit(0);
  if (err != hipSuccess) printf("[%s:%d]Failed to initialize HIP\n", __FILE__, __LINE__);
  err = hipGetDeviceCount(&count);
  if (err != hipSuccess) printf("[%s:%d]Failed to count HIP devices\n", __FILE__, __LINE__);
  err = hipDeviceGet(&dev, 0);
  if (err != hipSuccess) printf("[%s:%d]Failed to get HIP device\n", __FILE__, __LINE__);
  err = hipDeviceGetName(name, 256, dev);
  if (err != hipSuccess) printf("[%s:%d]Failed to get HIP device name\n", __FILE__, __LINE__);
  err = hipCtxCreate(&ctx, 0, dev);
  if (err != hipSuccess) printf("[%s:%d]Failed to create HIP context\n", __FILE__, __LINE__);
  err = hipMalloc(&dZ, nbytes);
  if (err != hipSuccess) printf("[%s:%d]Failed to malloc HIP device memory\n", __FILE__, __LINE__);
  err = hipMalloc(&dX, nbytes);
  if (err != hipSuccess) printf("[%s:%d]Failed to malloc HIP device memory\n", __FILE__, __LINE__);
  err = hipMalloc(&dY, nbytes);
  if (err != hipSuccess) printf("[%s:%d]Failed to malloc HIP device memory\n", __FILE__, __LINE__);
  err = hipModuleLoad(&mod, "./kernel.hip");
  if (err != hipSuccess) printf("[%s:%d]Failed to load HIP kernel\n", __FILE__, __LINE__);
  err = hipEventCreate(&start);
  if (err != hipSuccess) printf("[%s:%d]Failed to create HIP event\n", __FILE__, __LINE__);
  err = hipEventCreate(&stop);
  if (err != hipSuccess) printf("[%s:%d]Failed to create HIP event\n", __FILE__, __LINE__);
}

void Compute(int loop) {
  if (!func) {
    err = hipModuleGetFunction(&func, mod, "saxpy");
    if (err != hipSuccess) printf("[%s:%d]\n", __FILE__, __LINE__);
  }
  void* params[4] = { &dZ, &A, &dX, &dY };
  hipEventRecord(start, NULL);
  for (int i = 0; i < loop; i++)  {
    err = hipModuleLaunchKernel(func, 1, 1, 1, SIZE, 1, 1, 0, 0, params, NULL);
    if (err != hipSuccess) printf("[%s:%d]Failed to launch HIP kernel\n", __FILE__, __LINE__);
  }
  hipEventSynchronize(stop);
  //err = hipCtxSynchronize();
  //if (err != hipSuccess) {printf("[%s:%d]Failed to sync on HIP context\n", __FILE__, __LINE__); printf("error code: %d",err);};
}

void Finalize() {
  err = hipFree(dZ);
  if (err != hipSuccess) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = hipFree(dX);
  if (err != hipSuccess) printf("[%s:%d]\n", __FILE__, __LINE__);
  err = hipFree(dY);
  if (err != hipSuccess) printf("[%s:%d]\n", __FILE__, __LINE__);
  //err = hipModuleUnload(mod);
  if (err != hipSuccess) printf("[%s:%d]\n", __FILE__, __LINE__);
  hipCtxDestroy(ctx);
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  SIZE = atoi(argv[1]);
  LOOP = atoi(argv[2]);
  func = NULL;
  nbytes = SIZE * sizeof(float);
  REPEATS = 1;
  if (argc == 5) {
    REPEATS = atoi(argv[3]);
    LOGFILE = argv[4];
  }

  printf("[%s:%d] SIZE[%d][%luB] LOOP[%d]\n", __FILE__, __LINE__, SIZE, nbytes, LOOP);
  t0 = now();

  i0 = now();
  Init();
  i1 = now();

  for (int i = 0; i < REPEATS; ++ i) {
    c0 = now();
    Compute(LOOP);
    c1 = now();

    printf("latency [%lf] us\n", ((c1 - c0) / LOOP) * 1.e+6);
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

