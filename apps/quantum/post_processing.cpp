#include "iris/iris_openmp.h"

// qpu_i :  
// [ [64-bit to represent state, #count],  Example [0111101, 45]
// ]
// First record will have number of records 
extern "C" int qiree_post_processing(void *qpu1, void *qpu2, void *result)
{

    return IRIS_SUCCESS;
}

