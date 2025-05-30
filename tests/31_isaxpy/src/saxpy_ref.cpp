/**=============================================================================
Copyright (c) 2016 QUALCOMM Technologies Incorporated.
All Rights Reserved Qualcomm Proprietary
=============================================================================**/
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

void saxpy_ref(int32_t *Z,
               int32_t *X,
               int32_t *Y,
               int32_t A, 
               int32_t size)

{
    int i;
    for (i=0; i<size; i++) {
        Z[i] = A * X[i] + Y[i];
    }
}

