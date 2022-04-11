#ifndef IRIS_INCLUDE_IRIS_HOST2CUDA_H
#define IRIS_INCLUDE_IRIS_HOST2CUDA_H

#include <iris/iris_errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IRIS_HOST2CUDA_KERNEL_ARGS     size_t _off, size_t _ndr
#define IRIS_HOST2CUDA_KERNEL_BEGIN(i) for (i = _off; i < _off + _ndr; i++) {
#define IRIS_HOST2CUDA_KERNEL_END      }

#define __kernel
#define __global
#define __constant
#define __local
#define __restrict


#ifdef __cplusplus
extern "C" {
#endif

static pthread_mutex_t iris_host2cuda_mutex;
static int iris_host2cuda_kernel_idx;

void iris_host2cuda_init() {
  pthread_mutex_init(&iris_host2cuda_mutex, NULL);
}

void iris_host2cuda_finalize() {
  pthread_mutex_destroy(&iris_host2cuda_mutex);
}

static void iris_host2cuda_lock() {
  pthread_mutex_lock(&iris_host2cuda_mutex);
}

static void iris_host2cuda_unlock() {
  pthread_mutex_unlock(&iris_host2cuda_mutex);
}

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* IRIS_INCLUDE_IRIS_HOST2CUDA_H */

