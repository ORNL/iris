/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 

IRIS_CREATE_APIS(
        IRIS_FUNC(iris_, MATHOP_TYPE),
        IRIS_FUNC(iris_core_, MATHOP_TYPE),
        IRIS_FUNC(iris_task_, MATHOP_TYPE),
        IRIS_FUNC(iris_kernel_, MATHOP_TYPE), 1,
        NULL_OFFSET, GWS(size), NULL_LWS,
        OUT_TASK(out, DTYPE*, DTYPE, out, sizeof(DTYPE)*size),
        IN_TASK(in, DTYPE*, DTYPE, in, sizeof(DTYPE)*size),
        PARAM(size, size_t),
        CPP_TASK_TEMPLATE(IRIS_FUNC(iris_, CPP_MATHOP), MAP(DTYPE, DataType)),
        VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in[]))),
        VS_CODE_GEN(iris_cuda, CFUNC(IRIS_FUNC(iris_kernel_cuda_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in[]))),
        VS_CODE_GEN(iris_hip, CFUNC(IRIS_FUNC(iris_kernel_hip_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in[]))),
        VS_CODE_GEN(iris_cuda_host, INCLUDE(math.h, stdint.h, cuda.h), 
            EXPR(
                int threadsPerBlock = IRIS_MAX(256, size);
                int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
                //printf("Size: %lu seed:%ld p1:%f p2:%f stream:%p\n", size, seed, (float)p1, (float)p2, stream);
                if (stream != NULL) {
                    CUstream gpu_stream = reinterpret_cast<CUstream>(((void **)stream)[0]);
                    IRIS_FUNC(iris_kernel_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in, size);
                }
                )),
        VS_CODE_GEN(iris_hip_host, 
            INCLUDE(math.h, stdint.h, hip/hip_runtime_api.h), 
            EXPR(
                int threadsPerBlock = IRIS_MAX(256, size);
                int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
                //printf("Size: %lu seed:%ld p1:%f p2:%f stream:%p\n", size, seed, (float)p1, (float)p2, stream);
                if (stream != NULL) {
                    hipStream_t gpu_stream = reinterpret_cast<hipStream_t>(((void **)stream)[0]);
                    IRIS_FUNC(iris_kernel_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in, size);
                }
                )),
        TILED_CODE_GEN(FULL_SIZE(M), TILE_SIZE(tM), DIM(1), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled1d_,MATHOP_TYPE)),  
            ZIP(in,out)),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size() * col_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled2d_,MATHOP_TYPE)),  
            ZIP(in, out))
        );
#undef MATHOP_TYPE
#undef CPP_MATHOP
#undef CORE_MATHOP_FN



