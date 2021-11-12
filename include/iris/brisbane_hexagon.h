#ifndef BRISBANE_INCLUDE_BRISBANE_HEXAGON_H
#define BRISBANE_INCLUDE_BRISBANE_HEXAGON_H

#include <iris/brisbane_errno.h>
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

static pthread_mutex_t brisbane_hexagon_mutex;
static int brisbane_hexagon_kernel_idx;
static void* brisbane_hexagon_dl;

void brisbane_hexagon_init() {
  pthread_mutex_init(&brisbane_hexagon_mutex, NULL);
  brisbane_hexagon_dl = dlopen("libbrisbanehxg.so", RTLD_NOW);
  if (!brisbane_hexagon_dl) printf("[%s:%d] no libbrisbanehxg.so\n", __FILE__, __LINE__);
  rpcmem_init();
}

void brisbane_hexagon_finalize() {
  pthread_mutex_destroy(&brisbane_hexagon_mutex);
  if (brisbane_hexagon_dl) dlclose(brisbane_hexagon_dl);
  rpcmem_deinit();
}

static void brisbane_hexagon_lock() {
  pthread_mutex_lock(&brisbane_hexagon_mutex);
}

static void brisbane_hexagon_unlock() {
  pthread_mutex_unlock(&brisbane_hexagon_mutex);
}
void* brisbane_hexagon_rpcmem_alloc(int heapid, uint32_t flags, int size) {
  return rpcmem_alloc(heapid, flags, size);
}
void brisbane_hexagon_rpcmem_free(void* po) {
  rpcmem_free(po);
}

#endif /* BRISBANE_INCLUDE_BRISBANE_HEXAGON_H */

