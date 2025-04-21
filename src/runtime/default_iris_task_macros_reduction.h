/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 

IRIS_CREATE_APIS(
        IRIS_FUNC(iris_reduction_, REDUCTION),
        IRIS_FUNC(iris_reduction_core_, REDUCTION),
        IRIS_FUNC(iris_reduction_task_, REDUCTION),
        IRIS_FUNC(iris_reduction_kernel_, REDUCTION), 1,
        NULL_OFFSET, GWS(size), NULL_LWS,
        OUT_TASK(out, DTYPE*, DTYPE, out, sizeof(DTYPE)*1),
        IN_TASK(in, DTYPE*, DTYPE, in, sizeof(DTYPE)*size),
        PARAM(size, size_t),
        CPP_TASK_TEMPLATE(IRIS_FUNC(iris_reduction_, CPP_REDUCTION), MAP(DTYPE, DataType)),
        VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h, iris/default_iris_openmp_reduction.h), 
			EXPR(
			    IRIS_FUNC(iris_reduction_openmp_, CPP_REDUCTION)<DTYPE>(out, in, size);
			)),
        VS_CODE_GEN(iris_cuda_host, 
			INCLUDE(math.h, stdint.h, cuda.h, iris/default_iris_gpu_reduction.h), 
			EXPR(
                iris_reduction_gpu<DTYPE, IRIS_FUNC(iris_op_, CPP_REDUCTION)>(out, in, size, IRIS_FUNC(iris_op_, CPP_REDUCTION)(), stream);
			)),
        VS_CODE_GEN(iris_hip_host, 
			INCLUDE(math.h, stdint.h, hip/hip_runtime_api.h, iris/default_iris_gpu_reduction.h), 
			EXPR(
                iris_reduction_gpu<DTYPE, IRIS_FUNC(iris_op_, CPP_REDUCTION)>(out, in, size, IRIS_FUNC(iris_op_, CPP_REDUCTION)(), stream);
			)),
        TILED_CODE_GEN(FULL_SIZE(M), TILE_SIZE(tM), DIM(1), EXCLUDE(size),
            MAP(size, row_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_REDUCTION)), 
            CFUNC(IRIS_FUNC(iris_tiled1d_, REDUCTION)),  
            ZIP(in,out)),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(size),
            MAP(size, row_tile_size() * col_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_REDUCTION)), 
            CFUNC(IRIS_FUNC(iris_tiled2d_, REDUCTION)),  
            ZIP(in, out))
        );
#undef CPP_REDUCTION
#undef REDUCTION



