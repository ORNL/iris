#ifndef BRISBANE_INCLUDE_BRISBANE_OPENMP_H
#define BRISBANE_INCLUDE_BRISBANE_OPENMP_H

#include <brisbane/brisbane_errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BRISBANE_OPENMP_KERNEL_ARGS     size_t _off, size_t _ndr
#define BRISBANE_OPENMP_KERNEL_BEGIN(i) for (i = _off; i < _off + _ndr; i++) {
#define BRISBANE_OPENMP_KERNEL_END      }

#define __kernel
#define __global
#define __constant
#define __local
#define __restrict

static pthread_mutex_t brisbane_openmp_mutex;
static int brisbane_openmp_kernel_idx;

void brisbane_openmp_init() {
  pthread_mutex_init(&brisbane_openmp_mutex, NULL);
}

void brisbane_openmp_finalize() {
  pthread_mutex_destroy(&brisbane_openmp_mutex);
}

static void brisbane_openmp_lock() {
  pthread_mutex_lock(&brisbane_openmp_mutex);
}

static void brisbane_openmp_unlock() {
  pthread_mutex_unlock(&brisbane_openmp_mutex);
}

#endif /* BRISBANE_INCLUDE_BRISBANE_OPENMP_H */

