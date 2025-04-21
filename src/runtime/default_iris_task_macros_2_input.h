/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 
IRIS_CREATE_APIS(
        IRIS_FUNC(iris_, MATHOP_TYPE),
        IRIS_FUNC(iris_core_v_v_, MATHOP_TYPE),
        IRIS_FUNC(iris_task_v_v_, MATHOP_TYPE),
        IRIS_FUNC(iris_kernel_v_v_, MATHOP_TYPE), 1,
        NULL_OFFSET, GWS(size), NULL_LWS,
        OUT_TASK(out, DTYPE*, DTYPE, out, sizeof(DTYPE)*size),
        IN_TASK(in0, DTYPE*, DTYPE, in0, sizeof(DTYPE)*size),
        IN_TASK(in1, DTYPE*, DTYPE, in1, sizeof(DTYPE)*size),
        PARAM(size, size_t),
        CPP_TASK_TEMPLATE(IRIS_FUNC(iris_, CPP_MATHOP), MAP(DTYPE, DataType)),
        VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0[], in1[]))),
        VS_CODE_GEN(iris_cuda, CFUNC(IRIS_FUNC(iris_kernel_v_v_cuda_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0[], in1[]))),
        VS_CODE_GEN(iris_hip, CFUNC(IRIS_FUNC(iris_kernel_v_v_hip_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0[], in1[]))),
        VS_CODE_GEN(iris_cuda_host, INCLUDE(math.h, stdint.h, cuda.h), 
            EXPR(
                int threadsPerBlock = IRIS_MAX(256, size);
                int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
                //printf("Size: %lu seed:%ld p1:%f p2:%f stream:%p\n", size, seed, (float)p1, (float)p2, stream);
                if (stream != NULL) {
                    CUstream gpu_stream = reinterpret_cast<CUstream>(((void **)stream)[0]);
                    IRIS_FUNC(iris_kernel_v_v_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in0, in1, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_v_v_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in0, in1, size);
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
                    IRIS_FUNC(iris_kernel_v_v_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in0, in1, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_v_v_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in0, in1, size);
                }
                )),
        TILED_CODE_GEN(FULL_SIZE(M), TILE_SIZE(tM), DIM(1), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled1d_v_v_,MATHOP_TYPE)),  
            ZIP(in0,in1,out)),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size() * col_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled2d_v_v_,MATHOP_TYPE)),  
            ZIP(in0,in1,out))
        );

IRIS_CREATE_APIS(
        IRIS_FUNC(iris_, MATHOP_TYPE),
        IRIS_FUNC(iris_core_v_s_, MATHOP_TYPE),
        IRIS_FUNC(iris_task_v_s_, MATHOP_TYPE),
        IRIS_FUNC(iris_kernel_v_s_, MATHOP_TYPE), 1,
        NULL_OFFSET, GWS(size), NULL_LWS,
        OUT_TASK(out, DTYPE*, DTYPE, out, sizeof(DTYPE)*size),
        IN_TASK(in0, DTYPE*, DTYPE, in0, sizeof(DTYPE)*size),
        PARAM(in1, DTYPE),
        PARAM(size, size_t),
        CPP_TASK_TEMPLATE(IRIS_FUNC(iris_, CPP_MATHOP), MAP(DTYPE, DataType)),
        VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0[], in1))),
        VS_CODE_GEN(iris_cuda, CFUNC(IRIS_FUNC(iris_kernel_v_s_cuda_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0[], in1))),
        VS_CODE_GEN(iris_hip, CFUNC(IRIS_FUNC(iris_kernel_v_s_hip_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0[], in1))),
        VS_CODE_GEN(iris_cuda_host, INCLUDE(math.h, stdint.h, cuda.h), 
            EXPR(
                int threadsPerBlock = IRIS_MAX(256, size);
                int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
                //printf("Size: %lu seed:%ld p1:%f p2:%f stream:%p\n", size, seed, (float)p1, (float)p2, stream);
                if (stream != NULL) {
                    CUstream gpu_stream = reinterpret_cast<CUstream>(((void **)stream)[0]);
                    IRIS_FUNC(iris_kernel_v_s_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in0, in1, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_v_s_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in0, in1, size);
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
                    IRIS_FUNC(iris_kernel_v_s_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in0, in1, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_v_s_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in0, in1, size);
                }
                )),
        TILED_CODE_GEN(FULL_SIZE(M), TILE_SIZE(tM), DIM(1), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled1d_v_s_,MATHOP_TYPE)),  
            ZIP(in0,out)),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size() * col_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled2d_v_s_,MATHOP_TYPE)),  
            ZIP(in0,out))
        );

IRIS_CREATE_APIS(
        IRIS_FUNC(iris_, MATHOP_TYPE),
        IRIS_FUNC(iris_core_s_v_, MATHOP_TYPE),
        IRIS_FUNC(iris_task_s_v_, MATHOP_TYPE),
        IRIS_FUNC(iris_kernel_s_v_, MATHOP_TYPE), 1,
        NULL_OFFSET, GWS(size), NULL_LWS,
        OUT_TASK(out, DTYPE*, DTYPE, out, sizeof(DTYPE)*size),
        PARAM(in0, DTYPE),
        IN_TASK(in1, DTYPE*, DTYPE, in1, sizeof(DTYPE)*size),
        PARAM(size, size_t),
        CPP_TASK_TEMPLATE(IRIS_FUNC(iris_, CPP_MATHOP), MAP(DTYPE, DataType)),
        VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0, in1[]))),
        VS_CODE_GEN(iris_cuda, CFUNC(IRIS_FUNC(iris_kernel_s_v_cuda_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0, in1[]))),
        VS_CODE_GEN(iris_hip, CFUNC(IRIS_FUNC(iris_kernel_s_v_hip_, MATHOP_TYPE)), INCLUDE(math.h, stdint.h), EXPR(out[] = CORE_MATHOP_FN(in0, in1[]))),
        VS_CODE_GEN(iris_cuda_host, INCLUDE(math.h, stdint.h, cuda.h), 
            EXPR(
                int threadsPerBlock = IRIS_MAX(256, size);
                int blocksPerGrid = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);
                //printf("Size: %lu seed:%ld p1:%f p2:%f stream:%p\n", size, seed, (float)p1, (float)p2, stream);
                if (stream != NULL) {
                    CUstream gpu_stream = reinterpret_cast<CUstream>(((void **)stream)[0]);
                    IRIS_FUNC(iris_kernel_s_v_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in0, in1, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_s_v_cuda_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in0, in1, size);
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
                    IRIS_FUNC(iris_kernel_s_v_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock, 0, gpu_stream>>>(out, in0, in1, size);
                }
                else {
                    IRIS_FUNC(iris_kernel_s_v_hip_, MATHOP_TYPE)<<<blocksPerGrid, threadsPerBlock>>>(out, in0, in1, size);
                }
                )),
        TILED_CODE_GEN(FULL_SIZE(M), TILE_SIZE(tM), DIM(1), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled1d_s_v_,MATHOP_TYPE)),  
            ZIP(in1,out)),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(size),
            //INCLUDE(iris/zip.hpp, iris/iris.h),
            MAP(size, row_tile_size() * col_tile_size()),
            GENERIC(IRIS_FUNC(iris_tiled_, CPP_MATHOP)), 
            CFUNC(IRIS_FUNC(iris_tiled2d_s_v_,MATHOP_TYPE)),  
            ZIP(in1,out))
        );

#undef MATHOP_TYPE
#undef CPP_MATHOP
#undef CORE_MATHOP_FN

