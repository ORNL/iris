#include "iris/iris_openmp.h"
#include "iris/qiree.h"
#include <stdio.h>

// qpu_i :  
// [ [64-bit to represent state, #count],  Example [0111101, 45]
// ]
// First record will have number of records 
extern "C" int qiree_post_processing(CQireeResultRecord *qpu1, CQireeResultRecord *qpu2, void *result, IRIS_OPENMP_KERNEL_ARGS)
{
    printf("qpu1:%p result:%p \n", qpu1, result);
    printf("qpu2:%p result:%p \n", qpu2, result);
    printf("QPU1: %lu: %lu qpu1:%p\n", qpu1[0].key, qpu1[0].count, qpu1);
    printf("QPU1: %lu: %lu qpu1:%p\n", qpu1[1].key, qpu1[1].count, qpu1);
    printf("QPU2: %lu: %lu qpu2:%p\n", qpu2[0].key, qpu2[0].count, qpu2);
    return IRIS_SUCCESS;
}

