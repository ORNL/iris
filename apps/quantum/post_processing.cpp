#include "iris/iris_openmp.h"

extern "C" int qiree_post_processing(void *qpu1, void *qpu2, void *result)
{
    return IRIS_SUCCESS;
}
