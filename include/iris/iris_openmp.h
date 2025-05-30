#ifndef IRIS_INCLUDE_IRIS_OPENMP_H
#define IRIS_INCLUDE_IRIS_OPENMP_H

#include <iris/iris_errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//#define IRIS_OPENMP_KERNEL_ARGS     size_t _off, size_t _ndr
//#define IRIS_OPENMP_KERNEL_BEGIN(i) for (i = _off; i < _off + _ndr; i++) {
//#define IRIS_OPENMP_KERNEL_END      }
#define IRIS_OPENMP_KERNEL_ARGS     size_t *_off, size_t *_bws
#define IRIS_OPENMP_KERNEL_BEGIN(i) for (i = _off[0]; i < _off[0] + _bws[0]; i++) {
#define IRIS_OPENMP_KERNEL_END      }
#define IRIS_OPENMP_KERNEL_BEGIN2D(i, j) \
    for (i = _off[1]; i < _off[1]+_bws[1]; i++) { \
        for (j = _off[0]; j < _off[0]+_bws[0]; j++) {
#define IRIS_OPENMP_KERNEL_END2D      \
        } \
    } 

#define __kernel
#define __global
#define __constant
#define __local
#define __restrict

#ifdef __cplusplus
extern "C" {
#endif

static pthread_mutex_t iris_openmp_mutex;
static int iris_openmp_kernel_idx;

void iris_openmp_init() {
  pthread_mutex_init(&iris_openmp_mutex, NULL);
}

void iris_openmp_finalize() {
  pthread_mutex_destroy(&iris_openmp_mutex);
}

static void iris_openmp_lock() {
  pthread_mutex_lock(&iris_openmp_mutex);
}

static void iris_openmp_unlock() {
  pthread_mutex_unlock(&iris_openmp_mutex);
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* IRIS_INCLUDE_IRIS_OPENMP_H */

