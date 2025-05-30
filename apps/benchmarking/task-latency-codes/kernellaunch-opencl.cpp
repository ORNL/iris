#include "timer.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

double t0, t1;
double i0, i1;
double f0, f1;
double c0, c1;

int SIZE;
int LOOP;
size_t nbytes;
float A;

cl_platform_id plat;
cl_device_id dev;
cl_context ctx;
cl_command_queue stream;
cl_mem dZ, dX, dY;
cl_program mod;
cl_kernel func;
cl_int err;

void Init() {
  int count;
  char name[256];
  err = clGetPlatformIDs(1, &plat, NULL);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  printf("name[%s]\n", name);
  ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  dZ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &err);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  dX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &err);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  dY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nbytes, NULL, &err);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);

  stream = clCreateCommandQueue(ctx, dev, 0, &err);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);

  char* src;
  size_t srclen;
  err = read_file("kernel.cl", &src, &srclen);
  if (err != 1) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);

  mod = clCreateProgramWithSource(ctx, 1, (const char**) &src, (const size_t*) &srclen, &err);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clBuildProgram(mod, 1, &dev, NULL, NULL, NULL);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
}

void Compute(int loop) {
  if (loop == 1) {
    func = clCreateKernel(mod, "saxpy", &err);
    if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  }
  err = clSetKernelArg(func, 0, sizeof(cl_mem), &dZ);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clSetKernelArg(func, 1, sizeof(A), &A);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clSetKernelArg(func, 2, sizeof(cl_mem), &dX);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clSetKernelArg(func, 3, sizeof(cl_mem), &dY);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);

  size_t gws = 1;
  size_t lws = 1;
  for (int i = 0; i < loop; i++)  {
    err = clEnqueueNDRangeKernel(stream, func, 1, NULL, &gws, &lws, 0, NULL, NULL);
  }
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clFinish(stream);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
}

void Finalize() {
  err = clReleaseMemObject(dZ);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clReleaseMemObject(dX);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clReleaseMemObject(dY);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = clReleaseContext(ctx);
  if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  SIZE = atoi(argv[1]);
  LOOP = atoi(argv[2]);
  nbytes = SIZE * sizeof(float);
  printf("[%s:%d] SIZE[%d][%luB] LOOP[%d]\n", __FILE__, __LINE__, SIZE, nbytes, LOOP);

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

  printf("latency [%lf] us\n", ((c1 - c0) / LOOP) * 1.e+6);
  printf("secs: T[%lf] I[%lf] C[%lf] F[%lf]\n", t1 - t0, i1 - i0, c1 - c0, f1 - f0);

  return 0;
}

