#pragma once

#include "iris/iris.h"
#include "iris/iris_macros.h"

IRIS_TASK_APIS_CPP(
        isaxpy_cpp,      // C++ overload function API for both core and task
        isaxpy_core, // Function name for core API
        isaxpy_task, // Function name for task API
        "saxpy", 1,
        NULL_OFFSET, GWS(SIZE), NULL_LWS, 
        OUT_TASK(Z, int32_t *, int32_t, Z, sizeof(int32_t)*SIZE),
        IN_TASK(X, int32_t *, int32_t, X, sizeof(int32_t)*SIZE),
        IN_TASK(Y, int32_t *, int32_t, Y, sizeof(int32_t)*SIZE),
        PARAM(SIZE, int32_t),
        PARAM(A, int32_t),
        PARAM(cuUsecPtr, int32_t*, iris_dsp),
        PARAM(cuCycPtr, int32_t*, iris_dsp));
