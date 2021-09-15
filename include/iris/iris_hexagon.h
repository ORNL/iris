#ifndef IRIS_INCLUDE_IRIS_HEXAGON_H
#define IRIS_INCLUDE_IRIS_HEXAGON_H

#include <iris/iris_errno.h>
#include <iris/hexagon/rpcmem.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define __kernel
#define __global
#define __constant
#define __local
#define __restrict

static pthread_mutex_t iris_hexagon_mutex;
static int iris_hexagon_kernel_idx;
static void* iris_hexagon_dl;

void iris_hexagon_init() {
  pthread_mutex_init(&iris_hexagon_mutex, NULL);
  iris_hexagon_dl = dlopen("libirishxg.so", RTLD_NOW);
  if (!iris_hexagon_dl) printf("[%s:%d] no libirishxg.so\n", __FILE__, __LINE__);
  rpcmem_init();
}

void iris_hexagon_finalize() {
  pthread_mutex_destroy(&iris_hexagon_mutex);
  if (iris_hexagon_dl) dlclose(iris_hexagon_dl);
  rpcmem_deinit();
}

static void iris_hexagon_lock() {
  pthread_mutex_lock(&iris_hexagon_mutex);
}

static void iris_hexagon_unlock() {
  pthread_mutex_unlock(&iris_hexagon_mutex);
}
void* iris_hexagon_rpcmem_alloc(int heapid, uint32_t flags, int size) {
  return rpcmem_alloc(heapid, flags, size);
}
void iris_hexagon_rpcmem_free(void* po) {
  rpcmem_free(po);
}

#endif /* IRIS_INCLUDE_IRIS_HEXAGON_H */

