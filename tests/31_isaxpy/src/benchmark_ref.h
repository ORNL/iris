/**=============================================================================
Copyright (c) 2016, 2017 QUALCOMM Technologies Incorporated.
All Rights Reserved Qualcomm Proprietary
=============================================================================**/
#ifndef BENCHMARK_ASM_H
#define BENCHMARK_ASM_H

#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>

void saxpy_ref(int32_t *Z,
               int32_t *X,
               int32_t *Y,
               int32_t A, 
               int32_t size);

#ifdef __cplusplus
}
#endif

#endif
