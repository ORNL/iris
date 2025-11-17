#include "iris/iris_openmp.h"
#include "iris/qiree.h"
#include <stdio.h>

// qpu_i :  
// [ [64-bit to represent state, #count],  Example [0111101, 45]
// ]
// First record will have number of records 
extern "C" int qiree_post_processing(void *qpu1, void *result, IRIS_OPENMP_KERNEL_ARGS)
{
    CQireeResultRecord *r_qpu1 = (CQireeResultRecord *)qpu1;
    printf("qpu1:%p result:%p \n", qpu1, result);
    //CQireeResultRecord *r_qpu2 = (CQireeResultRecord *)qpu2;
    printf("QPU1: %lu: %lu qpu1:%p\n", r_qpu1[0].key, r_qpu1[0].count, qpu1);
    printf("QPU1: %lu: %lu qpu1:%p\n", r_qpu1[1].key, r_qpu1[1].count, qpu1);
    //printf("QPU2: %lu: %lu qpu2:%p\n", r_qpu2[0].key, r_qpu2[0].count, qpu2);
    return IRIS_SUCCESS;
}

