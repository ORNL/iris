/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 

IRIS_CREATE_APIS(
        IRIS_FUNC(iris_rope_, ROPE_TYPE),
        IRIS_FUNC(iris_core_rope_, ROPE_TYPE),
        IRIS_FUNC(iris_task_rope_, ROPE_TYPE),
        IRIS_FUNC(iris_kernel_rope_, ROPE_TYPE), 1,
        NULL_OFFSET, GWS(cols, rows), NULL_LWS,
        IN_OUT_TASK(inout, DTYPE*, DTYPE, inout, sizeof(DTYPE)*rows*cols*2),
        PARAM(cols, size_t),
        PARAM(rows, size_t),
        PARAM(pos, int),
        PARAM(theta_scale, DTYPE),
        CPP_TASK_TEMPLATE(IRIS_FUNC(iris_rope_, CPP_ROPE), MAP(DTYPE, DataType)),
        VS_CODE_GEN(iris_cpu_host, INCLUDE(math.h, stdlib.h, stdint.h, iris/default_iris_openmp_ai.h), 
            EXPR(
                iris_rope_cpu<DTYPE>(inout, cols, rows, pos, theta_scale)
                )),
        VS_CODE_GEN(iris_cuda_host, INCLUDE(math.h, stdint.h, cuda.h, iris/default_iris_gpu_ai.h), 
            EXPR(
                iris_rope_gpu<DTYPE>(inout, cols, rows, pos, theta_scale, stream)
                )),
        VS_CODE_GEN(iris_hip_host, 
            INCLUDE(math.h, stdint.h, hip/hip_runtime_api.h, iris/default_iris_gpu_ai.h), 
            EXPR(
                iris_rope_gpu<DTYPE>(inout, cols, rows, pos, theta_scale, stream)
                )),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), 
                                        EXCLUDE(rows,cols),
            MAP(cols, col_tile_size()),
            MAP(rows, row_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_rope_, CPP_ROPE)), 
            CFUNC(IRIS_FUNC(iris_tiled_rope_, ROPE_TYPE)),  
            ZIP(inout))
        );
#undef ROPE_TYPE
#undef CPP_ROPE



