#
# Author : Narasinga Rao Miniskar
# Date   : Jun 16 2022
# Contact for bug : miniskarnr@ornl.gov
#

cmake_minimum_required(VERSION 3.5)

# Sources(C) in sd_hexagon file without extension 
#set(SDHEXAGON_SOURCES
#   )

# OpenMP kernel Sources in src directory with extension
#set(OPENMP_KERNEL_SOURCES
#   )

# Host2OpenCL kernel Sources in src directory with extension
#set(HOST2OPENCL_KERNEL_SOURCES
#   )

# Host2XILINX kernel Sources in src directory with extension
#set(HOST2XILINX_KERNEL_SOURCES
#   )

# Host2CUDA kernel Sources in src directory with extension
#set(HOST2CUDA_KERNEL_SOURCES
#   )

# Host2HIP kernel Sources in src directory with extension
#set(HOST2HIP_KERNEL_SOURCES
#   )

# Hexagon kernel Sources in src directory with extension
#set(HEXAGON_KERNEL_SOURCES
#   )

# Benchmark CUDA Sources in src directory with extension
#set(CUDA_KERNEL_SOURCES
#   )

# Benchmark SPV Sources in src directory with extension
#set(SPV_SOURCES
#   )

# Benchmark HIP Sources in src directory with extension
#set(HIP_KERNEL_SOURCES
#   )

# IRIS Hexagon Interface Sources
#set(IRIS_HEXAGON_INTERFACE_SOURCES
#   )

# Benchmark application executable Sources in src directory with extension
#set(APP_SOURCES
#   )

# Benchmark application library Sources in src directory with extension
#set(APP_LIB_SOURCES
#   )

# Name of executable
#set(APP_EXECUTABLE  sobel3x3)

project(IRISKernels)
include(GNUInstallDirs)

set(HEXAGON_RELATIVE_STUB "hexagon")
set(IRISKernels_VERSION_MAJOR 1)
set(IRISKernels_VERSION_MINOR 0)
set(IRISKernels_VERSION_PATCH 0)

set(IRISKernels_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(IRISKernels_INC_DIR ${PROJECT_SOURCE_DIR}/include)
set(IRISKernels_BIN_DIR ${PROJECT_BINARY_DIR}/bin)

option(NO_OPENMP "Disable OpenMP in CPU kernel"    OFF)
option(USE_RISCV "Use RISCV toolchain "    OFF )
message(STATUS "******* CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT USE_RISCV)
find_package(CUDAToolkit)
endif()
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
  # Support for NVPACK
  set(CUDAToolkit_TARGET_NAME "armv7-linux-androideabi")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  set(CUDAToolkit_TARGET_NAME "armv7-linux-gnueabihf")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  if(ANDROID_ARCH_NAME STREQUAL "arm64")
    set(CUDAToolkit_TARGET_NAME "aarch64-linux-androideabi")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CUDAToolkit_TARGET_NAME "aarch64-qnx")
  else()
    set(CUDAToolkit_TARGET_NAME "aarch64-linux")
  endif(ANDROID_ARCH_NAME STREQUAL "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  set(CUDAToolkit_TARGET_NAME "ppc64le-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(CUDAToolkit_TARGET_NAME "x86_64-linux")
endif()

#message(STATUS "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}")
#message(STATUS "CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")
#message(STATUS "CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
#message(STATUS "CUDAToolkit_BIN_DIR=${CUDAToolkit_BIN_DIR}")
#message(STATUS "CUDAToolkit_TARGET_NAME=${CUDAToolkit_TARGET_NAME}")
#message(STATUS "CUDAToolkit_TARGET_DIR=${CUDAToolkit_TARGET_DIR}")

set(AUTO_CUDA_INCLUDE )
set(AUTO_CUDA_LDFLAGS )
if (CUDAToolkit_FOUND)
set(CUDAToolkit_ROOT_DIR ${CUDAToolkit_TARGET_DIR})
set(CUSTOM_CUDAToolkit_TARGET_DIR "${CUDAToolkit_ROOT_DIR}/targets/${CUDAToolkit_TARGET_NAME}")
set(CUDAToolkit_MATH_LIB_DIR "${CUDAToolkit_TARGET_DIR}/../../math_libs")
#message(STATUS "CUSTOM_CUDAToolkit_TARGET_DIR=${CUSTOM_CUDAToolkit_TARGET_DIR}")
#message(STATUS "CUDAToolkit_MATH_LIB_DIR=${CUDAToolkit_MATH_LIB_DIR}")
set(AUTO_CUDA_INCLUDE "-I${CUDAToolkit_TARGET_DIR}/include -I${CUSTOM_CUDAToolkit_TARGET_DIR}/include")
set(AUTO_CUDA_LDFLAGS "-L${CUDAToolkit_LIBRARY_DIR} -L${CUDAToolkit_MATH_LIB_DIR}/lib64 -L${CUDAToolkit_MATH_LIB_DIR}/lib")
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_TARGET_DIR}")
#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CXX_COMPILER g++)
message(STATUS "Host C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Host C++ compiler: ${CMAKE_CXX_COMPILER}")
else()
message(STATUS "CUDA Toolkit not found")
endif()

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" ${CMAKE_MODULE_PATH})
if (NOT USE_RISCV)
find_package(HIP)
endif() # USE_RISCV
set(AUTO_HIP_INCLUDE )
set(AUTO_HIP_LDFLAGS )
if (HIP_ROOT_DIR)
set(AUTO_HIP_INCLUDE "-I${HIP_ROOT_DIR}/../hipsolver/include -I${HIP_ROOT_DIR}/../hipfft/include -I${HIP_ROOT_DIR}/../hipblas/include -I${HIP_ROOT_DIR}/../rocblas/include -I${HIP_ROOT_DIR}/include")
set(AUTO_HIP_LDFLAGS "-L${HIP_ROOT_DIR}/../hipblas/lib64 -L${HIP_ROOT_DIR}/../hipblas/lib -L${HIP_ROOT_DIR}/../rocblas/lib64 -L${HIP_ROOT_DIR}/../rocblas/lib -L${HIP_ROOT_DIR}/../hipfft/lib64 -L${HIP_ROOT_DIR}/../hipfft/lib -L${HIP_ROOT_DIR}/../hipsolver/lib64 -L${HIP_ROOT_DIR}/../hipsolver/lib -L${HIP_ROOT_DIR}/lib64 -L${HIP_ROOT_DIR}/lib")
message(STATUS "HIP Toolkit found in HIP_ROOT_DIR=${HIP_ROOT_DIR}")
#message(STATUS "AUTO_HIP_LDFLAGS=${AUTO_HIP_LDFLAGS}")
#message(STATUS "AUTO_HIP_INCLUDE=${AUTO_HIP_INCLUDE}")
else()
message(STATUS "HIP not found")
endif()

if (HIP_ROOT_DIR)
    set(HIP_ENABLED ON)
else()
    set(HIP_ENABLED OFF)
endif()
option(ENABLE_OPENMP_NVCC "Enable Host cuda kernel with NVC++ for OpenMP offload"    OFF)
option(ENABLE_HOST2CUDA_NATIVE "Compile host2cuda kernels with NVC++ compiler"    OFF)
option(ENABLE_HOST2HIP_NATIVE "Compile host2hip kernels with hipcc compiler"    OFF)
option(APP_LIB_STATIC "Enable static app lib"    OFF)
option(SNAPDRAGON "Enable support for Hexagon"    OFF )
option(CDSP_FLAG "Enable support for CDSP"     OFF )
option(ADSP_FLAG "Enable support for ADSP"     OFF )
option(MDSP_FLAG "Enable support for MDSP"     OFF )
option(SLPI_FLAG "Enable support for SLPI"     OFF )
option(ENABLE_STD11 "Enable STD 11"    ON)
option(USE_SPIR "Use SPIR toolchain "    OFF )
if (SNAPDRAGON)
option(USE_HEXAGON "Use HEXAGON toolchain "    ON )
else()
option(USE_HEXAGON "Use HEXAGON toolchain "    OFF )
endif()
option(USE_XILINX "Use Xilinx toolchain "    OFF )
option(USE_HIP "Use HIP toolchain "    ${HIP_ENABLED})
option(USE_CUDA "Use CUDA toolchain "    ${CUDAToolkit_FOUND})
option(USE_AARCH64 "Use Aarch64 toolchain "    OFF )
option(USE_NDK   "Use NDK toolchain "          OFF )
option(USE_PYTHON  "Enable support for Python" ON )
option(USE_FORTRAN "Enable support for Fortran 90" OFF)

option(USE_SIGHANDLER   "Enable signal handler" OFF)
option(USE_HUB          "Enable hub"            OFF)

set(LOCAL_COMMON_FLAGS "-I$ENV{IRIS}/utils -I. -I${CMAKE_BINARY_DIR} -I$ENV{IRIS}/include")

if (USE_RISCV)
set(CMAKE_C_COMPILER   riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
endif() # USE_RISCV

function(initVar var_name default_value)
#    message(STATUS "args var:(${var_name}) default_value:(${default_value})")
    if (NOT ${var_name})
        #message(STATUS "var  (${var_name}) is not defined. Setting default: ${default_value}")
        set(${var_name} "${default_value}" PARENT_SCOPE)
    endif() #var_name
    #message(STATUS "var  (${var_name}) is set to ${default_value}")
endfunction() # initVar
initVar(APP_ENVNAME "IRIS_APP")
initVar(OPENMP_FLAGS "")
initVar(OPENMP_LDFLAGS "")
initVar(CUDA_FLAGS "")
initVar(HIP_FLAGS "")
initVar(HOST2CUDA_FLAGS "")
initVar(HOST2CUDA_LDFLAGS "")
initVar(HOST2CUDA_COMP_FLAGS "")
initVar(NVCC_CU cc70)
if (ENABLE_OPENMP_NVCC)
set(HOST2CUDA_COMP_COMPILER nvc++)
set(OPENMP_GPU_FLAGS -mp=gpu -gpu=${NVCC_CU})
set(OPENMP_GPU_LDFLAGS )
else()
initVar(HOST2CUDA_COMP_COMPILER clang++)
set(OPENMP_GPU_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
set(OPENMP_GPU_LDFLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
endif()
if (ENABLE_HOST2CUDA_NATIVE)
set(HOST2CUDA_COMP_COMPILER nvc++)
set(OPENMP_GPU_FLAGS "")
set(OPENMP_GPU_LDFLAGS "")
endif()
initVar(HOST2CUDA_COMP_LDFLAGS "")
initVar(HOST2HIP_FLAGS "")
initVar(HOST2HIP_LDFLAGS "")
initVar(HOST2ROC_FLAGS "")
initVar(HOST2ROC_LDFLAGS "")
initVar(HOST2XILINX_FLAGS "")
initVar(HOST2XILINX_LDFLAGS "")
initVar(HOST2OPENCL_FLAGS "")
initVar(HOST2OPENCL_LDFLAGS "")
initVar(APP_FLAGS "")
initVar(APP_LDFLAGS "")
initVar(HEXAGON_FLAGS "")
initVar(HEXAGON_LDFLAGS "")

#set(COMMON_FLAGS "-Dirishxg_free_stub=irishxg_free_stub_core -Dirishxg_alloc_stub=irishxg_alloc_stub_core")
#set(COMMON_FLAGS "")
#set(STUB_FLAG "")
initVar(COMMON_FLAGS "")
initVar(LIB_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
initVar(BIN_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
initVar(KERNEL_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
initVar(XILINX_FLAGS "")
initVar(XILINX_LDFLAGS "-O3")

if (IS_ABSOLUTE ${KERNEL_INSTALL_PATH})
set(KERNEL_INSTALL_PATH_DIR ${KERNEL_INSTALL_PATH})
else()
set(KERNEL_INSTALL_PATH_DIR "\$${APP_ENVNAME}/${KERNEL_INSTALL_PATH}")
endif()

function(install_iris_files fpattern install_prefix)
    file(GLOB_RECURSE folder_files ${fpattern})
    foreach(file ${folder_files})
        get_filename_component(dir ${file} DIRECTORY)
        if (NOT PRESERVE_OPENCL_INSTALL_DIRECTORIES)
            install(FILES ${file} DESTINATION ${install_prefix}/)
        else() #PRESERVE_OPENCL_INSTALL_DIRECTORIES
            string(REPLACE ${PROJECT_SOURCE_DIR}/src/ "" dir ${dir})
            install(FILES ${file} DESTINATION ${install_prefix}/${dir})
        endif()
    endforeach()
endfunction() #install_iris_files


if (CDSP_FLAG) 
set(DSP_TAG cdsp)
set(AUTOVEC_FLAG "ENABLE_AUTOVEC=1")
set(STUB_FLAG "CDSP_FLAG=1")
endif()
if (ADSP_FLAG) 
set(DSP_TAG adsp)
set(AUTOVEC_FLAG "")
set(STUB_FLAG "ADSP_FLAG=1")
endif()
if (MDSP_FLAG) 
set(DSP_TAG mdsp)
set(AUTOVEC_FLAG "")
set(STUB_FLAG "MDSP_FLAG=1")
endif()
if (SLPI_FLAG) 
set(DSP_TAG sdsp)
set(AUTOVEC_FLAG "")
set(STUB_FLAG "SLPI_FLAG=1")
endif()
if (NOT STUB_FLAG)
set(DSP_TAG cdsp)
set(STUB_FLAG "CDSP_FLAG=1")
set(AUTOVEC_FLAG "ENABLE_AUTOVEC=1")
endif()
set(STUB_LIB_NAME "irishxg")
set(LINK_LIB_DSPRPC adsprpc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable")
if (USE_NDK)
set(STUB_TARGET "android_Release_aarch64")
else() #USE_NDK
set(STUB_TARGET "UbuntuARM_Release_aarch64")
endif() #USE_NDK

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -std=gnu99")
if (ENABLE_STD11)
if (CMAKE_VERSION VERSION_LESS "3.1")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)
endif() #ENABLE_STD11

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HEXAGON_TARGET "hexagon_Release_dynamic_toolv83_v66")
set(ANDROID_STUB_LIBDIR "${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/${STUB_TARGET}")
set(HEXAGON_CORE_LIBDIR "${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/${HEXAGON_TARGET}")
set(ANDROIDHXG      "${ANDROID_STUB_LIBDIR}/lib${STUB_LIB_NAME}.so")
set(ANDROIDHXG_SKEL "${ANDROID_STUB_LIBDIR}/lib${STUB_LIB_NAME}_skel.so")

set(OPENMP_INC "")
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(OPENMP_INC "")
else()
if (NOT NO_OPENMP)
    set(OPENMP_INC "-fopenmp")
endif()
endif()
set(HOST2XILINX_FLAGS "${HOST2XILINX_FLAGS} -I$ENV{XILINX_HLS}/include -I$ENV{XILINX_VITIS}/include -I$ENV{XILINX_XRT}/include/ -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I. ")
set(HOST2OPENCL_FLAGS "${HOST2OPENCL_FLAGS} -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I.")
set(HOST2CUDA_FLAGS "${HOST2CUDA_FLAGS} ${AUTO_CUDA_INCLUDE} -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I.")
set(HOST2CUDA_COMP_FLAGS "${HOST2CUDA_COMP_FLAGS} ${AUTO_CUDA_INCLUDE} -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I.")
set(HOST2HIP_FLAGS "${HOST2HIP_FLAGS} ${AUTO_HIP_INCLUDE} -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I.")
set(HOST2ROC_FLAGS "${HOST2ROC_FLAGS} -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I.")
set(OPENMP_FLAGS "${OPENMP_FLAGS} -DENABLE_IRIS ${OPENMP_INC} -O3 -g -fPIC ${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS} -I.")
set(HOST2XILINX_LDFLAGS "${HOST2XILINX_LDFLAGS} ${OPENMP_INC} -g -fPIC")
set(HOST2OPENCL_LDFLAGS "${HOST2OPENCL_LDFLAGS} ${OPENMP_INC} -g -fPIC")
set(HOST2CUDA_LDFLAGS "${HOST2CUDA_LDFLAGS} ${AUTO_CUDA_LDFLAGS} ${OPENMP_INC} -g -fPIC")
set(HOST2CUDA_COMP_LDFLAGS "${HOST2CUDA_COMP_LDFLAGS} ${AUTO_CUDA_LDFLAGS} ${OPENMP_INC} -g -fPIC")
set(HOST2HIP_LDFLAGS "${HOST2HIP_LDFLAGS} ${AUTO_HIP_LDFLAGS} ${OPENMP_INC} -g -fPIC")
set(OPENMP_LDFLAGS "${OPENMP_LDFLAGS} ${OPENMP_INC} -g -fPIC")
set(HEXAGON_FLAGS "${HEXAGON_FLAGS} -fPIC -g -O3 ${COMMON_FLAGS}  ${LOCAL_COMMON_FLAGS} -I$ENV{IRIS}/include/iris/hexagon")
set(HEXAGON_LDFLAGS "${HEXAGON_LDFLAGS} -g -L${ANDROID_STUB_LIBDIR}  -Wl,--no-as-needed")
set(APP_LDFLAGS "${APP_LDFLAGS} -L$ENV{IRIS}/lib64 -L$ENV{IRIS}/lib -L${ANDROID_STUB_LIBDIR}/")
set(APP_FLAGS "${APP_FLAGS} ${COMMON_FLAGS} -O3 -fPIC -g  ${LOCAL_COMMON_FLAGS} -I.")
set(APP_FLAGS "${APP_FLAGS} -DENABLE_IRIS")
set(SKEL_CSRCS_STR "")
if (USE_HEXAGON)
set(APP_FLAGS "${APP_FLAGS} -DENABLE_HEXAGON")
endif()
set(APP_FLAGS "${APP_FLAGS} ${OPENMP_FLAGS}")
set(APP_LDFLAGS "${APP_LDFLAGS} ${OPENMP_LDFLAGS}")
if (SNAPDRAGON)
set(OPENMP_FLAGS "${OPENMP_FLAGS} -fvectorize")
set(OPENMP_LDFLAGS "${OPENMP_LDFLAGS} -L${ANDROID_STUB_LIBDIR} ")
set(APP_FLAGS "${APP_FLAGS} -I$ENV{HEXAGON_SDK_ROOT}/incs/stddef/")
set(APP_FLAGS "${APP_FLAGS} -DSNAPDRAGON")
set(HEXAGON_RPC "$ENV{HEXAGON_SDK_ROOT}/libs/common/rpcmem/${STUB_TARGET}/ship/rpcmem.a")
endif() #SNAPDRAGON
if (SDHEXAGON_SOURCES)
set(SKEL_CSRCS_STR "SKEL_CSRCS=\"${SDHEXAGON_SOURCES}\"")
endif() # SDHEXAGON_SOURCES
if (USE_RISCV)
set(OPENMP_FLAGS "${OPENMP_FLAGS}  -DDISABLE_DYNAMIC_LINKING -fopenmp  -DOPENMP -static")
set(OPENMP_LDFLAGS "${OPENMP_LDFLAGS} -static -lgomp -lpthread -ldl")
endif() # USE_RISCV


function(hexagonTargets)
   set(IRIS_HEXAGON_INTERFACE_SOURCE "${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/iris_hexagon_interface.c")
   if (NOT IRIS_HEXAGON_INTERFACE_SOURCES)
   set(IRIS_HEXAGON_INTERFACE_SOURCES "${IRIS_HEXAGON_INTERFACE_SOURCE}")
   endif()
   #message(STATUS "IRIS_HEXAGON_INTERFACE_SOURCE: ${IRIS_HEXAGON_INTERFACE_SOURCE}")
   add_custom_command(
     OUTPUT ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/Makefile
     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} 
     COMMAND mkdir -p ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}
     COMMAND mkdir -p ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/sd_host
     COMMAND mkdir -p ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/sd_hexagon
     COMMAND cp -f ${HEXAGON_KERNEL_SOURCES}  ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/sd_hexagon/. 
     COMMAND rm -f ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/irishxg.idl
     COMMAND cp -f ${PROJECT_SOURCE_DIR}/irishxg.idl ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/irishxg.idl
     COMMAND rm -f ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/Makefile
     COMMAND cp -f $ENV{IRIS}/utils/Makefile.hexagon ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/Makefile
     )
   add_custom_command(
     OUTPUT ${IRIS_HEXAGON_INTERFACE_SOURCE}
     COMMAND rm -f ${IRIS_HEXAGON_INTERFACE_SOURCE}
     COMMAND cp -f $ENV{IRIS}/include/iris/hexagon/iris_interface.h  ${IRIS_HEXAGON_INTERFACE_SOURCE}
     )
   add_custom_command(
     OUTPUT ${ANDROIDHXG}
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
     COMMAND cp -f $ENV{IRIS}/utils/android.min ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/android.min
     COMMAND cp -f $ENV{IRIS}/utils/android_deps.min ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/android_deps.min
     COMMAND cp -f $ENV{IRIS}/include/iris/hexagon/stub_imp.h ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/sd_host/stub.c
     COMMAND make -C ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB} tree_clean V=${STUB_TARGET} ${STUB_FLAG} DSP_TAG=${DSP_TAG} IRIS=$ENV{IRIS} VERBOSE=1 API_LEVEL=28 -f $ENV{IRIS}/utils/Makefile.hexagon
     COMMAND make -C ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB} tree V=${STUB_TARGET} ${STUB_FLAG} DSP_TAG=${DSP_TAG} IRIS=$ENV{IRIS} VERBOSE=1 API_LEVEL=28  -f $ENV{IRIS}/utils/Makefile.hexagon
     BYPRODUCTS ${CMAKE_BINARY_DIR}/${STUB_TARGET}
     )
   add_custom_command(
     OUTPUT ${ANDROIDHXG_SKEL}
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
     COMMAND cp -f $ENV{IRIS}/include/iris/hexagon/std_kernels.h ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/sd_hexagon/std_kernels.c
     COMMAND cp -f $ENV{IRIS}/utils/hexagon.min ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/hexagon.min
     COMMAND cp -f $ENV{IRIS}/utils/hexagon_deps.min ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/hexagon_deps.min
     COMMAND make -C ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB} tree_clean V=${HEXAGON_TARGET} ${AUTOVEC_FLAG} IRIS=$ENV{IRIS} VERBOSE=1 API_LEVEL=28 ${SKEL_CSRCS_STR} -f $ENV{IRIS}/utils/Makefile.hexagon
     COMMAND make -C ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB} tree V=${HEXAGON_TARGET} ${AUTOVEC_FLAG} IRIS=$ENV{IRIS} VERBOSE=1 API_LEVEL=28 ${SKEL_CSRCS_STR} -f $ENV{IRIS}/utils/Makefile.hexagon
     BYPRODUCTS ${CMAKE_BINARY_DIR}/${HEXAGON_TARGET}
     )
   #message(STATUS "\n ANDROIDHXG_SKEL : ${ANDROIDHXG_SKEL}\n")
   #message(STATUS "\n ANDROIDHXG: ${ANDROIDHXG}\n")
   #message(STATUS "\n CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}\n")
   add_custom_command(
     OUTPUT ${CMAKE_BINARY_DIR}/output0x15bd6ed/testsig-0x15bd6ed.so 
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
     COMMAND echo 'y' | python $ENV{HEXAGON_SDK_ROOT}/tools/elfsigner/elfsigner.py -zz -t 0x15bd6ed -o output0x15bd6ed
     )
   add_custom_command(
     OUTPUT ${CMAKE_BINARY_DIR}/output0x4061ec60/testsig-0x4061ec60.so 
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
     COMMAND echo 'y' | python $ENV{HEXAGON_SDK_ROOT}/tools/elfsigner/elfsigner.py -zz -t 0x4061ec60 -o output0x4061ec60
     )
   add_custom_target(irishxg_makefile DEPENDS ${CMAKE_BINARY_DIR}/${HEXAGON_RELATIVE_STUB}/Makefile)
   add_custom_target(irishxg_interface DEPENDS ${IRIS_HEXAGON_INTERFACE_SOURCE})
   add_custom_target(irishxg_lib DEPENDS ${ANDROIDHXG})
   add_custom_target(irishxg_skel DEPENDS ${ANDROIDHXG_SKEL})
   add_custom_target(hexagon_elfsigner ALL DEPENDS ${CMAKE_BINARY_DIR}/output0x4061ec60/testsig-0x4061ec60.so ${CMAKE_BINARY_DIR}/output0x15bd6ed/testsig-0x15bd6ed.so)
   add_dependencies(irishxg_skel irishxg_makefile)
   add_dependencies(irishxg_interface irishxg_makefile)
   add_dependencies(irishxg_lib irishxg_makefile irishxg_skel)
   add_library(kernel.hexagon SHARED ${IRIS_HEXAGON_INTERFACE_SOURCES})

   add_dependencies(kernel.hexagon hexagon_elfsigner irishxg_makefile iris_app_cpu_dsp_interface.h irishxg_interface irishxg_skel irishxg_lib)
   set_target_properties(kernel.hexagon PROPERTIES COMPILE_FLAGS ${HEXAGON_FLAGS})
   set_target_properties(kernel.hexagon PROPERTIES LINK_FLAGS ${HEXAGON_LDFLAGS})
   set_target_properties(kernel.hexagon PROPERTIES PREFIX "")
   target_link_libraries(kernel.hexagon ${HEXAGON_RPC})
   target_link_libraries(kernel.hexagon ${STUB_LIB_NAME})
   if (USE_NDK)
   target_link_libraries(kernel.hexagon log)
   endif() # USE_NDK
   install(TARGETS kernel.hexagon DESTINATION ${LIB_INSTALL_PATH})
   install(FILES ${CMAKE_BINARY_DIR}/output0x15bd6ed/testsig-0x15bd6ed.so DESTINATION ${LIB_INSTALL_PATH})
   install(FILES ${CMAKE_BINARY_DIR}/output0x4061ec60/testsig-0x4061ec60.so DESTINATION ${LIB_INSTALL_PATH})
   install(DIRECTORY ${ANDROID_STUB_LIBDIR}/ship/ DESTINATION ${CMAKE_INSTALL_PREFIX}/ FILES_MATCHING PATTERN "*")
   install(DIRECTORY ${HEXAGON_CORE_LIBDIR}/ship/  DESTINATION ${CMAKE_INSTALL_PREFIX}/ FILES_MATCHING PATTERN "*")
endfunction() #hexagonTargets

function(extractIRISAPIs)
   set(src_include_paths "")
   set(src_rules "")
   set(src_iris_files "")
   list(APPEND list_of_sources ${ARGN})
   #message(STATUS "\n ARG : ${list_of_sources}\n")
   while(list_of_sources)
        list(GET list_of_sources 0 first_core)
        #message(STATUS "\n FIRST ARG : ${first_core}\n")
        list(REMOVE_AT list_of_sources 0)
        string(REGEX REPLACE " $" "" first "${first_core}")
        string(REGEX REPLACE "/" ".__" rule_name "${first}.iris")
        string(REGEX REPLACE "\n$" "" rule_name "${rule_name}")
        get_filename_component(SRC_PATH ${first} DIRECTORY)
        if (SRC_PATH STREQUAL "")
        set(SRC_PATH ".")
        endif()
        #message(STATUS "\n Processing file: ${first} in path:${SRC_PATH}\n")
        get_filename_component(SRC_DIR_PATH ${SRC_PATH} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
        get_filename_component(SRC_FILE_PATH ${first} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
        list(APPEND src_include_paths "-I${SRC_DIR_PATH}")
        add_custom_command(OUTPUT  ${CMAKE_BINARY_DIR}/${rule_name}.iris 
            COMMAND python3 $ENV{IRIS}/utils/extract_iris_task_calls.py -input ${SRC_FILE_PATH} -compile_options "${APP_FLAGS}" -output ${CMAKE_BINARY_DIR}/${rule_name}.iris -extract 
            DEPENDS ${SRC_FILE_PATH}
        )
        #message(STATUS " RULE: ${rule_name}")
        add_custom_target(${rule_name} DEPENDS ${CMAKE_BINARY_DIR}/${rule_name}.iris)
        #message(STATUS "\n Processing file: ${first} in path:${SRC_PATH} rule:${rule_name}\n")
        list(APPEND src_rules "${rule_name}")
        list(APPEND src_iris_files "${CMAKE_BINARY_DIR}/${rule_name}.iris")
   endwhile()
   add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/iris_app_cpu_dsp_interface.h
           COMMAND python3 $ENV{IRIS}/utils/extract_iris_task_calls.py -input ${src_iris_files} -output ${CMAKE_BINARY_DIR}/iris_app_cpu_dsp_interface.h -generate $ENV{IRIS_NO_CODEGEN}
           DEPENDS ${src_rules}
           )
    add_custom_target(iris_app_cpu_dsp_interface.h DEPENDS ${CMAKE_BINARY_DIR}/iris_app_cpu_dsp_interface.h)
endfunction()
if (EXTRACT_IRIS_KERNEL_SIGNATURES)
extractIRISAPIs(${EXTRACT_IRIS_KERNEL_SIGNATURES})
endif()
if (NOT ANALYZE_IRIS_LIBRARY_SOURCES_ONLY)
if (APP_SOURCES OR APP_LIB_SOURCES)
extractIRISAPIs(${APP_SOURCES} ${APP_LIB_SOURCES})
endif()
else()
if (APP_LIB_SOURCES)
extractIRISAPIs(${APP_LIB_SOURCES})
endif()
endif()

if (SNAPDRAGON AND USE_HEXAGON AND HEXAGON_KERNEL_SOURCES)
hexagonTargets()
endif() #SNAPDRAGON

if (SNAPDRAGON)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.android.source
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
  COMMAND echo "export PATH=\\$$PWD:/system/bin:\\$$PATH" > ${CMAKE_BINARY_DIR}/setup.android.source 
  COMMAND echo "export LD_LIBRARY_PATH=.:\\$$PWD:\\$$PWD/lib64:\\$$PWD/lib:\\$$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.android.source 
  )
add_custom_target(setup.android.source ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.android.source)
install(FILES ${CMAKE_BINARY_DIR}/setup.android.source DESTINATION ${CMAKE_INSTALL_PREFIX})
else() #SNAPDRAGON
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.source
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "SCRIPT_FILE=\${BASH_SOURCE[0]}" > ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "PYTHON_EXE=python" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "version=\$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "if [[ -z \"\$version\" ]]" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "then" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    PYTHON_EXE=python3" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    version=\$(python3 -V 2>&1 | grep -Po '(?<=Python )(.+)')" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    if [[ -z \"\$version\" ]]" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    then" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    echo \"No Python is detected! Cannot source\" " >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    return" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    fi" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "fi" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "SCRIPT=$(\$PYTHON_EXE -c \"import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))\" \${SCRIPT_FILE})" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "export ${APP_ENVNAME}=$(dirname -- \${SCRIPT})" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export PATH=\$${APP_ENVNAME}/bin:\$PATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export IRIS_KERNEL_DIR=${KERNEL_INSTALL_PATH_DIR}" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export LD_LIBRARY_PATH=.:\$${APP_ENVNAME}/lib64:\$${APP_ENVNAME}/lib:\$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export PYTHONPATH=\$${APP_ENVNAME}/scripts:\$PYTHONPATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export PKG_CONFIG_PATH=\$${APP_ENVNAME}/lib64/pkg_config:\$${APP_ENVNAME}/lib/pkg_config:\$PKG_CONFIG_PATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"*********************** ${APP_ENVNAME} Environent *********************\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"${APP_ENVNAME}: \$${APP_ENVNAME}\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"*************************************************************\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "if [[ -d \"\$${APP_ENVNAME}/lib64\" ]] " >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "then" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    export EMCONFIG_PATH=\$${APP_ENVNAME}/lib64" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "else" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    if [[ -d \"\$${APP_ENVNAME}/lib\" ]] " >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    then" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "        export EMCONFIG_PATH=\$${APP_ENVNAME}/lib" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "    fi" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "fi" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  )
add_custom_target(setup.source ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.source)
install(FILES ${CMAKE_BINARY_DIR}/setup.source DESTINATION ${CMAKE_INSTALL_PREFIX})
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.host.source
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
  COMMAND echo "export PATH=\\$$PWD/bin:\\$$PWD:\\$$PATH" > ${CMAKE_BINARY_DIR}/setup.host.source 
  COMMAND echo "export LD_LIBRARY_PATH=.:\\$$PWD:\\$$PWD/lib64:\\$$PWD/lib:\\$$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.host.source 
  )
add_custom_target(setup.host ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.host.source)
install(FILES ${CMAKE_BINARY_DIR}/setup.host.source DESTINATION ${CMAKE_INSTALL_PREFIX})
endif() #SNAPDRAGON

if (USE_CUDA)
if (CUDA_KERNEL_SOURCES)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/kernel.ptx
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} 
  COMMAND cat ${CUDA_KERNEL_SOURCES} > ${CMAKE_CURRENT_BINARY_DIR}/__iris_kernel.cu
  COMMAND nvcc -ptx ${CMAKE_CURRENT_BINARY_DIR}/__iris_kernel.cu ${CUDA_FLAGS} -o ${CMAKE_BINARY_DIR}/kernel.ptx --output-directory ${CMAKE_BINARY_DIR}
  BYPRODUCTS ${CMAKE_BINARY_DIR}/kernel.ptx
  )
add_custom_target(kernel.ptx ALL DEPENDS ${CMAKE_BINARY_DIR}/kernel.ptx)
add_dependencies(kernel.ptx iris_app_cpu_dsp_interface.h)
install(FILES ${CMAKE_BINARY_DIR}/kernel.ptx DESTINATION ${KERNEL_INSTALL_PATH})
endif() # CUDA_KERNEL_SOURCES
endif() # USE_CUDA

if (USE_XILINX)
if (NOT XILINX_TARGET AND DEFINED ENV{XCL_EMULATION_MODE})
set(XILINX_TARGET  "$ENV{XCL_EMULATION_MODE}")  # sw_emu, hw_emu, hw
else()
initVar(XILINX_TARGET "hw")
endif() # XILINX_TARGET
message(STATUS "XILINX Target (XCL_EMULATION_MODE): ${XILINX_TARGET}")
initVar(XILINX_LANGUAGE "opencl")
initVar(XILINX_VERSION "1")
initVar(XILINX_OUTPUT "xclbin")
#initVar(XILINX_FREQUENCY 300000000)
initVar(XILINX_DESIGN_CONFIG "${CMAKE_BINARY_DIR}/design.cfg")
message(STATUS "XILINX Design config (XILINX_DESIGN_CONFIG): ${XILINX_DESIGN_CONFIG}")
initVar(XTMP_DIR "${CMAKE_BINARY_DIR}/_x/${XILINX_TARGET}/${XILINX_LANGUAGE}/${XILINX_VERSION}")
initVar(XILINX_XCLBIN "kernel-fpga.${XILINX_OUTPUT}")
if (NOT XILINX_PLATFORM)
execute_process(COMMAND bash -c "platforminfo  -l | grep baseName | awk '{print $2}' | head -n1 | sed -e s/,//g  -e s/\\\"//g"   OUTPUT_VARIABLE XILINX_PLATFORM)
string(REGEX REPLACE "\n$" "" XILINX_PLATFORM "${XILINX_PLATFORM}")
endif()
message(STATUS "XILINX_PLATFORM: ${XILINX_PLATFORM}")

function(xilinx_compile)
   set(all_xos "")
   set(all_kernel_names "")
   set(all_connectivities "")
   list(APPEND list_of_pairs ${ARGN})
    #message(STATUS "\n ARG : ${list_of_pairs}\n")
   set(XILINX_FLAGS ${XILINX_FLAGS} "-I$ENV{XILINX_VITIS}/include" "-I$ENV{XILINX_XRT}/include/" "--optimize" "3")
   if (XILINX_SPLIT_KERNEL)
    message(STATUS "XILINX_SPLIT_DESIGN: ${XILINX_SPLIT_DESIGN}")
    get_filename_component(XILINX_DESIGN_CONFIG_ABSOLUTE ${XILINX_SPLIT_DESIGN} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
    add_custom_command(
           OUTPUT ${XILINX_DESIGN_CONFIG}
           COMMAND sed -e "s/XILINX_PLATFORM/${XILINX_PLATFORM}/g" ${XILINX_DESIGN_CONFIG_ABSOLUTE} > ${XILINX_DESIGN_CONFIG}
           )
   else() # XILINX_SPLIT_KERNEL
   if (EXISTS "${PROJECT_SOURCE_DIR}/design.cfg")
    add_custom_command(
      OUTPUT ${XILINX_DESIGN_CONFIG}
      COMMAND cp -f ${PROJECT_SOURCE_DIR}/design.cfg ${XILINX_DESIGN_CONFIG}
    )
   else() # design.cfg
   add_custom_command(
           OUTPUT ${XILINX_DESIGN_CONFIG}
           COMMAND echo 'platform=${XILINX_PLATFORM}'  > ${XILINX_DESIGN_CONFIG}
           COMMAND echo 'debug=1' >> ${XILINX_DESIGN_CONFIG} 
           COMMAND echo 'profile_kernel=data:all:all:all' >> ${XILINX_DESIGN_CONFIG}
           COMMAND echo 'save-temps=1' >> ${XILINX_DESIGN_CONFIG}
           )
   endif() # design.cfg
   endif() # XILINX_SPLIT_KERNEL
   add_custom_target(design.cfg DEPENDS ${XILINX_DESIGN_CONFIG})
   set(COMMON_FLAGS_X  "${COMMON_FLAGS} ${LOCAL_COMMON_FLAGS}")
   string(REPLACE " " ";" COMMON_FLAGS_X ${COMMON_FLAGS_X})
   while(list_of_pairs)
        list(GET list_of_pairs 0 first)
        list(REMOVE_AT list_of_pairs 0)
        list(GET list_of_pairs 0 second)
        list(REMOVE_AT list_of_pairs 0)
        list(GET list_of_pairs 0 third)
        list(REMOVE_AT list_of_pairs 0)
        string(REGEX REPLACE "::.*" "" file_path "${third}")
        string(REGEX REPLACE ".*::" "" macros    "${third}")
        if (file_path STREQUAL macros)
        set(macros "")
        else()
        string(REPLACE "\"" "" macros ${macros})
        string(REPLACE " " ";" macros ${macros})
        endif()
        #message(STATUS "XILINX File:${file_path}")
        #message(STATUS "XILINX macros:${macros}")
        get_filename_component(SRC_PATH ${file_path} DIRECTORY)
        set(XILINX_FREQUENCY_ARG "")
        if (XILINX_FREQUENCY)
        set(XILINX_FREQUENCY_ARG "--hls.clock" "${XILINX_FREQUENCY}:${first}")
        endif()
        set(XILINX_VIVADO_JOBS_ARG "")
        if (XILINX_VIVADO_JOBS)
        set(XILINX_VIVADO_JOBS_ARG "--vivado.synth.jobs" "${XILINX_VIVADO_JOBS}" "--vivado.impl.jobs" "${XILINX_VIVADO_JOBS}")
        endif()
        add_custom_command(OUTPUT  ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/xo/${file_path}-${first}.xo
            COMMAND v++ -c -t ${XILINX_TARGET} --kernel ${first} --temp_dir ${XTMP_DIR} --platform ${XILINX_PLATFORM} --config ${XILINX_DESIGN_CONFIG} ${XILINX_FLAGS} ${COMMON_FLAGS_X} -I${PROJECT_SOURCE_DIR}/${SRC_PATH}  ${XILINX_FREQUENCY_ARG} ${XILINX_VIVADO_JOBS_ARG} ${macros} -I. ${PROJECT_SOURCE_DIR}/${file_path} -o ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/xo/${file_path}-${first}.xo
        )
        set(KNAME "k_${first}")
        add_custom_target(${KNAME} DEPENDS design.cfg ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/xo/${file_path}-${first}.xo)
        list(APPEND all_kernel_names "${KNAME}")
        if (NOT XILINX_SPLIT_KERNEL)
        list(APPEND all_connectivities "--connectivity.nk=${first}:1:${second}")
        endif()
        list(APPEND all_xos "${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/xo/${file_path}-${first}.xo")
        #message(STATUS "\n ARG : ${first} == ${second} == ${file_path}\n")
   endwhile()
   add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/${XILINX_XCLBIN}
           COMMAND v++ -l -t ${XILINX_TARGET} --temp_dir ${XTMP_DIR} --config ${XILINX_DESIGN_CONFIG} --platform ${XILINX_PLATFORM} -I. ${XILINX_LDFLAGS} ${all_connectivities} ${all_xos}  -o ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/${XILINX_XCLBIN}
           DEPENDS ${all_kernel_names}
           )
    add_custom_target(${XILINX_XCLBIN} ALL DEPENDS design.cfg ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/${XILINX_XCLBIN})
    install(FILES ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/${XILINX_XCLBIN} DESTINATION ${LIB_INSTALL_PATH}/${XILINX_TARGET}/${XILINX_LANGUAGE})
    install(FILES ${CMAKE_BINARY_DIR}/${XILINX_TARGET}/${XILINX_LANGUAGE}/${XILINX_XCLBIN} DESTINATION ${LIB_INSTALL_PATH}/)
endfunction()

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/emconfig.json
  COMMAND emconfigutil --platform ${XILINX_PLATFORM} --nd 1
  )
add_custom_target(emconfig.json ALL DEPENDS ${CMAKE_BINARY_DIR}/emconfig.json)
install(FILES ${CMAKE_BINARY_DIR}/emconfig.json DESTINATION ${LIB_INSTALL_PATH}/)
if (XILINX_OPENCL_SOURCES AND ${XILINX_LANGUAGE} STREQUAL "opencl")
xilinx_compile(${XILINX_OPENCL_SOURCES})
#add_dependencies(${XILINX_XCLBIN} emconfig.json)
endif() # XILINX_OPENCL_SOURCES
if (XILINX_KERNEL_SOURCES AND ${XILINX_LANGUAGE} STREQUAL "xilinx")
xilinx_compile(${XILINX_KERNEL_SOURCES})
#add_dependencies(${XILINX_XCLBIN} emconfig.json)
endif() # XILINX_KERNEL_SOURCES
endif() # USE_XILINX

if (USE_SPIR)
if (SPV_SOURCES)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/kernel.spv
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} 
  COMMAND clang -cc1 -finclude-default-header -triple spir ${SPV_SOURCES} -O0 -flto -emit-llvm-bc -o ${CMAKE_BINARY_DIR}/kernel.bc
  COMMAND llvm-spirv ${CMAKE_BINARY_DIR}/kernel.bc -o ${CMAKE_BINARY_DIR}/kernel.spv
  BYPRODUCTS ${CMAKE_BINARY_DIR}/kernel.bc ${CMAKE_BINARY_DIR}/kernel.spv
  )
add_custom_target(kernel.spv DEPENDS ${CMAKE_BINARY_DIR}/kernel.spv)
install(FILES ${CMAKE_BINARY_DIR}/kernel.spv DESTINATION ${KERNEL_INSTALL_PATH})
endif() # SPV_SOURCES
endif() # USE_SPIR

if (USE_HIP)
if (HIP_KERNEL_SOURCES)
if (HIP_FLAGS)
string(REPLACE " " ";" hip_flags ${HIP_FLAGS})
endif()
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/kernel.hip
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} 
  COMMAND cat ${HIP_KERNEL_SOURCES} > ${CMAKE_CURRENT_BINARY_DIR}/__iris_kernel.hip.cpp
  COMMAND hipcc --genco ${HIP_FLAGS} -o ${CMAKE_BINARY_DIR}/kernel.hip ${CMAKE_CURRENT_BINARY_DIR}/__iris_kernel.hip.cpp
  BYPRODUCTS ${CMAKE_BINARY_DIR}/kernel.hip
  )
add_custom_target(kernel.hip DEPENDS ${CMAKE_BINARY_DIR}/kernel.hip)
add_dependencies(kernel.hip iris_app_cpu_dsp_interface.h)
install(FILES ${CMAKE_BINARY_DIR}/kernel.hip DESTINATION ${KERNEL_INSTALL_PATH})
endif() #HIP_KERNEL_SOURCES
endif() # USE_HIP

function(addKernelRuleStatic kernel_target kernel_sources flags ldflags libraries)
        add_library(${kernel_target} STATIC ${kernel_sources})
        set_target_properties(${kernel_target} PROPERTIES COMPILE_FLAGS ${flags})
        set_target_properties(${kernel_target} PROPERTIES LINK_FLAGS ${ldflags})
        set_target_properties(${kernel_target} PROPERTIES PREFIX "")
        add_dependencies(${kernel_target} iris_app_cpu_dsp_interface.h)
        install(TARGETS ${kernel_target} DESTINATION ${LIB_INSTALL_PATH})
        if (libraries)
            target_link_libraries(${kernel_target} ${libraries})
        endif()
endfunction() #addKernelRuleStatic

function(addKernelRule kernel_target kernel_sources flags ldflags libraries)
        add_library(${kernel_target} SHARED ${kernel_sources})
        set_target_properties(${kernel_target} PROPERTIES COMPILE_FLAGS ${flags})
        set_target_properties(${kernel_target} PROPERTIES LINK_FLAGS ${ldflags})
        set_target_properties(${kernel_target} PROPERTIES PREFIX "")
        add_dependencies(${kernel_target} iris_app_cpu_dsp_interface.h)
        install(TARGETS ${kernel_target} DESTINATION ${LIB_INSTALL_PATH})
        if (libraries)
            target_link_libraries(${kernel_target} ${libraries})
        endif()
endfunction() #addKernelRule

function(addKernelSpecificCompiler 
                           kernel_target 
                           compiler
                           kernel_sources 
                           flags 
                           ldflags 
                           libraries 
                           output_obj_files 
                           output_rules)
        set(flags_X ${flags})
        set(ldflags_X ${ldflags})
        string(REPLACE " " ";" flags_X ${flags_X})
        string(REPLACE " " ";" ldflags_X ${ldflags_X})
        set(local_output_obj_files )
        foreach(cuda_source ${kernel_sources})
            message(STATUS ${cuda_source})
            string(REGEX REPLACE " $" "" first "${cuda_source}")
            string(REGEX REPLACE "/" ".__" rule_name "${first}.iris.${kernel_target}.specific")
            string(REGEX REPLACE "\n$" "" rule_name "${rule_name}")
            get_filename_component(SRC_PATH ${first} DIRECTORY)
            if (SRC_PATH STREQUAL "")
            set(SRC_PATH ".")
            endif()
            message(STATUS "\n Processing file: ${CMAKE_CURRENT_BINARY_DIR} ${first} in path:${SRC_PATH} rule:${rule_name}\n")
            get_filename_component(SRC_DIR_PATH ${SRC_PATH} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
            get_filename_component(SRC_FILE_PATH ${first} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
            message(STATUS "\n ${SRC_DIR_PATH} ${SRC_FILE_PATH}\n")
            set(output "${rule_name}.o")
            add_custom_command(
                    OUTPUT ${output}
                    COMMAND ${compiler} -O2 ${OPENMP_GPU_FLAGS} -g -O0 -c ${SRC_FILE_PATH} ${flags_X} -I$ENV{IRIS}/include -I${SRC_DIR_PATH} -o ${output}
                    DEPENDS ${SRC_FILE_PATH} iris_app_cpu_dsp_interface.h
                    VERBATIM
            )
            add_custom_target(${rule_name} DEPENDS ${output})
            list(APPEND local_output_obj_files ${output})
            list(APPEND output_rules ${rule_name})
        endforeach()
        set(libpaths)
        foreach(libname ${libraries})
        list(APPEND libpaths "-l${libname}")
        endforeach()
        message(STATUS "OBJ files: ${local_output_obj_files}")
        set(output_obj_files ${local_output_obj_files} PARENT_SCOPE)
        add_custom_command(
                OUTPUT ${kernel_target}.so
                COMMAND ${compiler} -O0 -g -fPIC -shared -Wl,-soname,${kernel_target}.so -o ${kernel_target}.so ${local_output_obj_files} ${ldflags_X} ${libpaths}  -Wl,-rpath,${LIB_INSTALL_PATH}
                DEPENDS ${output_rules}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
                )
        add_custom_target(${kernel_target} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${kernel_target}.so)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${kernel_target}.so DESTINATION ${LIB_INSTALL_PATH})
endfunction() #addKernelRule

function(addKernelCUDARule kernel_target 
                           kernel_sources 
                           flags 
                           ldflags 
                           libraries 
                           output_obj_files 
                           output_rules)
        set(flags_X ${flags})
        set(ldflags_X ${ldflags})
        string(REPLACE " " ";" flags_X ${flags_X})
        string(REPLACE " " ";" ldflags_X ${ldflags_X})
        set(local_output_obj_files )
        foreach(cuda_source ${kernel_sources})
            message(STATUS ${cuda_source})
            string(REGEX REPLACE " $" "" first "${cuda_source}")
            string(REGEX REPLACE "/" ".__" rule_name "${first}.iris")
            string(REGEX REPLACE "\n$" "" rule_name "${rule_name}")
            get_filename_component(SRC_PATH ${first} DIRECTORY)
            if (SRC_PATH STREQUAL "")
            set(SRC_PATH ".")
            endif()
            message(STATUS "\n Processing file: ${CMAKE_CURRENT_BINARY_DIR} ${first} in path:${SRC_PATH} rule:${rule_name}\n")
            get_filename_component(SRC_DIR_PATH ${SRC_PATH} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
            get_filename_component(SRC_FILE_PATH ${first} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})
            message(STATUS "\n ${SRC_DIR_PATH} ${SRC_FILE_PATH}\n")
            set(output "${rule_name}.o")
            add_custom_command(
                    OUTPUT ${output}
                    COMMAND ${HOST2CUDA_COMP_COMPILER} -O2 -c ${SRC_FILE_PATH} ${flags_X} -I$ENV{IRIS}/include -I${SRC_DIR_PATH} -o ${output}
                    DEPENDS ${SRC_FILE_PATH} iris_app_cpu_dsp_interface.h
                    VERBATIM
            )
            string(REGEX REPLACE "/" ".__" rule_name "${file_c}.iris")
            string(REGEX REPLACE "\n$" "" rule_name "${rule_name}")
            add_custom_target(${rule_name} DEPENDS ${output})
            list(APPEND local_output_obj_files ${output})
            list(APPEND output_rules ${rule_name})
        endforeach()
        set(libpaths)
        foreach(libname ${libraries})
        list(APPEND libpaths "-l${libname}")
        endforeach()
        message(STATUS "OBJ files: ${local_output_obj_files}")
        set(output_obj_files ${local_output_obj_files} PARENT_SCOPE)
        add_custom_command(
                OUTPUT lib${kernel_target}.a
                COMMAND llvm-ar rcs lib${kernel_target}.a ${local_output_obj_files} 
                DEPENDS ${output_rules}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
                )
        add_custom_target(${kernel_target} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib${kernel_target}.a)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib${kernel_target}.a DESTINATION ${LIB_INSTALL_PATH})
endfunction() #addKernelRule

if (OPENMP_KERNEL_SOURCES)
    if (USE_RISCV)
        addKernelRuleStatic(kernel.openmp "${OPENMP_KERNEL_SOURCES}" ${OPENMP_FLAGS} ${OPENMP_LDFLAGS} "${OPENMP_LIBRARIES}")
    else()
        addKernelRule(kernel.openmp "${OPENMP_KERNEL_SOURCES}" ${OPENMP_FLAGS} ${OPENMP_LDFLAGS} "${OPENMP_LIBRARIES}")
    endif()
    add_dependencies(kernel.openmp iris_app_cpu_dsp_interface.h)
    if (USE_NDK)
        target_link_libraries(kernel.openmp log)
    endif() # USE_NDK
endif() #OPENMP_KERNEL_SOURCES

if (HOST2OPENCL_KERNEL_SOURCES)
addKernelRule(kernel.host2opencl "${HOST2OPENCL_KERNEL_SOURCES}" ${HOST2OPENCL_FLAGS} ${HOST2OPENCL_LDFLAGS} "${HOST2OPENCL_LIBRARIES}")
endif() #HOST2OPENCL_KERNEL_SOURCES

if (HOST2XILINX_KERNEL_SOURCES AND USE_XILINX)
addKernelRule(kernel.host2opencl.xilinx "${HOST2XILINX_KERNEL_SOURCES}" ${HOST2XILINX_FLAGS} ${HOST2XILINX_LDFLAGS} "${HOST2XILINX_LIBRARIES}")
endif() #HOST2XILINX_KERNEL_SOURCES

set(output_obj_files "")
if (HOST2CUDA_COMP_KERNEL_SOURCES AND USE_CUDA)
set(output_rules "")
addKernelCUDARule(kernel.host2cudaComp 
                 "${HOST2CUDA_COMP_KERNEL_SOURCES}" 
                 "${HOST2CUDA_COMP_FLAGS}" 
                 "${HOST2CUDA_COMP_LDFLAGS}" 
                 "${HOST2CUDA_COMP_LIBRARIES}" 
                 "${output_obj_files}"
                 "${output_rules}")
#message(STATUS "Output obj files in CUDA: ${output_obj_files}")
set(HOST2CUDA_LDFLAGS "${HOST2CUDA_LDFLAGS} ${OPENMP_GPU_LDFLAGS} ${output_obj_files} -L${CMAKE_CURRENT_BINARY_DIR}")
set(HOST2CUDA_LIBRARIES  ${HOST2CUDA_LIBRARIES} )
#set(HOST2CUDA_LIBRARIES  "${HOST2CUDA_LIBRARIES} -lkernel.host2cudaComp")
endif() #HOST2CUDA_COMP_KERNEL_SOURCES
#message(STATUS "Output obj files in CUDA ALL: ${output_obj_files}")

if (HOST2CUDA_KERNEL_SOURCES AND USE_CUDA)
if (ENABLE_HOST2CUDA_NATIVE)
set(output_obj_files "")
set(output_rules "")
set(HOST2CUDA_FLAGS "${HOST2CUDA_FLAGS} -cuda -x cu -DNATIVE_CUDA")
set(HOST2CUDA_LDFLAGS "${HOST2CUDA_LDFLAGS} -cuda ")
addKernelSpecificCompiler(kernel.host2cuda nvc++ "${HOST2CUDA_KERNEL_SOURCES}" ${HOST2CUDA_FLAGS} ${HOST2CUDA_LDFLAGS} "${HOST2CUDA_LIBRARIES}" "${output_obj_files}" "${output_rules}")
set(HOST2CUDA_LDFLAGS "${HOST2CUDA_LDFLAGS} -stdpar ${OPENMP_GPU_LDFLAGS} ${output_obj_files} -lcudart -L${CMAKE_CURRENT_BINARY_DIR}")
set(HOST2CUDA_LIBRARIES  ${HOST2CUDA_LIBRARIES} )
else()
addKernelRule(kernel.host2cuda "${HOST2CUDA_KERNEL_SOURCES}" ${HOST2CUDA_FLAGS} ${HOST2CUDA_LDFLAGS} "${HOST2CUDA_LIBRARIES}")
endif() #HOST2CUDA_KERNEL_SOURCES
endif() #HOST2CUDA_KERNEL_SOURCES

if (HOST2HIP_KERNEL_SOURCES AND USE_HIP)
if (ENABLE_HOST2HIP_NATIVE)
set(output_obj_files "")
set(output_rules "")
addKernelSpecificCompiler(kernel.host2hip hipcc "${HOST2HIP_KERNEL_SOURCES}" ${HOST2HIP_FLAGS} ${HOST2HIP_LDFLAGS} "${HOST2HIP_LIBRARIES}" "${output_obj_files}" "${output_rules}")
set(HOST2HIP_LDFLAGS "${HOST2HIP_LDFLAGS} ${OPENMP_GPU_LDFLAGS} ${output_obj_files} -L${CMAKE_CURRENT_BINARY_DIR}")
set(HOST2HIP_LIBRARIES  ${HOST2HIP_LIBRARIES} )
else()
addKernelRule(kernel.host2hip "${HOST2HIP_KERNEL_SOURCES}" ${HOST2HIP_FLAGS} ${HOST2HIP_LDFLAGS} "${HOST2HIP_LIBRARIES}")
endif()
endif() #HOST2HIP_KERNEL_SOURCES

if (HOST2ROC_KERNEL_SOURCES AND USE_HIP)
addKernelRule(kernel.host2roc "${HOST2ROC_KERNEL_SOURCES}" ${HOST2ROC_FLAGS} ${HOST2ROC_LDFLAGS} "${HOST2ROC_LIBRARIES}")
endif() #HOST2ROC_KERNEL_SOURCES

#printf "0x%x\n" `adb shell cat /sys/devices/soc0/serial_number` > serial
install_iris_files("${PROJECT_SOURCE_DIR}/src/*.cl" ${KERNEL_INSTALL_PATH})
##install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION ${KERNEL_INSTALL_PATH}/   FILES_MATCHING PATTERN "*.cl")



if (APP_LIB_SOURCES)
    initVar(APP_LIBRARY_NAME "irisapp")
    if (APP_LIB_STATIC)
    add_library(${APP_LIBRARY_NAME} STATIC ${APP_LIB_SOURCES})
    else() # APP_LIB_STATIC
    add_library(${APP_LIBRARY_NAME} SHARED ${APP_LIB_SOURCES})
    endif() # APP_LIB_STATIC
    add_dependencies(${APP_LIBRARY_NAME} iris_app_cpu_dsp_interface.h)
    set_property(TARGET ${APP_LIBRARY_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
    if (SNAPDRAGON)
        if (USE_HEXAGON AND HEXAGON_KERNEL_SOURCES)
            add_dependencies(${APP_LIBRARY_NAME} kernel.hexagon)
            add_dependencies(${APP_LIBRARY_NAME} irishxg_lib)
            add_dependencies(${APP_LIBRARY_NAME} irishxg_skel)
            target_link_libraries(${APP_LIBRARY_NAME} ${STUB_LIB_NAME} ${LINK_LIB_DSPRPC})
        endif() #HEXAGON_KERNEL_SOURCES
        target_link_libraries(${APP_LIBRARY_NAME} iris log m)
        add_dependencies(${APP_LIBRARY_NAME} setup.android.source)
    else() # SNAPDRAGON
        target_link_libraries(${APP_LIBRARY_NAME} iris pthread dl m)
        add_dependencies(${APP_LIBRARY_NAME} setup.host)
    endif() #SNAPDRAGON
    if (USE_CUDA)
        if (CUDA_KERNEL_SOURCES)
            add_dependencies(${APP_LIBRARY_NAME} kernel.ptx)
        endif() #CUDA_KERNEL_SOURCES
    endif() #USE_CUDA
    if (USE_HIP)
        if (HIP_KERNEL_SOURCES)
            add_dependencies(${APP_LIBRARY_NAME} kernel.hip)
        endif() # HIP_KERNEL_SOURCES
    endif() # USE_HIP
    if (USE_RISCV)
    set(APP_LDFLAGS "${APP_LDFLAGS} -static -lgomp -lpthread -ldl")
    endif() # USE_RISCV
    set_target_properties(${APP_LIBRARY_NAME} PROPERTIES COMPILE_FLAGS ${APP_FLAGS})
    set_target_properties(${APP_LIBRARY_NAME} PROPERTIES LINK_FLAGS ${APP_LDFLAGS})
    if (NOT APP_LIB_NO_INSTALL)
    install(TARGETS ${APP_LIBRARY_NAME} DESTINATION ${LIB_INSTALL_PATH})
    endif() # APP_LIB_NO_INSTALL
endif() #APP_LIB_SOURCES

if (APP_SOURCES)
    initVar(APP_EXECUTABLE "app")
    add_executable(${APP_EXECUTABLE} ${APP_SOURCES})
    add_dependencies(${APP_EXECUTABLE} iris_app_cpu_dsp_interface.h)
    if (APP_LIB_SOURCES)
        target_link_libraries(${APP_EXECUTABLE} ${APP_LIBRARY_NAME})
    else() #APP_LIB_SOURCES
        if (SNAPDRAGON)
            if (USE_HEXAGON AND HEXAGON_KERNEL_SOURCES)
                add_dependencies(${APP_EXECUTABLE} kernel.hexagon)
                add_dependencies(${APP_EXECUTABLE} irishxg_lib)
                add_dependencies(${APP_EXECUTABLE} irishxg_skel)
                #message(STATUS "STUB LINK: ${STUB_LIB_NAME}")
                target_link_libraries(${APP_EXECUTABLE} ${STUB_LIB_NAME} ${LINK_LIB_DSPRPC})
            endif() #HEXAGON_KERNEL_SOURCES
            target_link_libraries(${APP_EXECUTABLE} iris log m)
            add_dependencies(${APP_EXECUTABLE} setup.android.source)
        else() # SNAPDRAGON
            target_link_libraries(${APP_EXECUTABLE} iris pthread dl m)
            add_dependencies(${APP_EXECUTABLE} setup.host)
        endif() #SNAPDRAGON
        if (USE_CUDA)
            if (CUDA_KERNEL_SOURCES)
                add_dependencies(${APP_EXECUTABLE} kernel.ptx)
            endif() #CUDA_KERNEL_SOURCES
        endif() #USE_CUDA
        if (USE_HIP)
            if (HIP_KERNEL_SOURCES)
                add_dependencies(${APP_EXECUTABLE} kernel.hip)
            endif() # HIP_KERNEL_SOURCES
        endif() # USE_HIP
    endif() #APP_LIB_SOURCES
    if (USE_RISCV)
        set(APP_LDFLAGS "${APP_LDFLAGS} -static -lgomp -lpthread -ldl -Wl,--unresolved-symbols=ignore-in-object-files")
    endif() # USE_RISCV
    set_target_properties(${APP_EXECUTABLE} PROPERTIES COMPILE_FLAGS ${APP_FLAGS})
    set_target_properties(${APP_EXECUTABLE} PROPERTIES LINK_FLAGS ${APP_LDFLAGS})
    install(TARGETS ${APP_EXECUTABLE} DESTINATION ${BIN_INSTALL_PATH})
endif() #APP_SOURCES

if (APP_SOURCES AND USE_XILINX)
if (XILINX_OPENCL_SOURCES OR XILINX_KERNEL_SOURCES)
add_dependencies(${APP_EXECUTABLE} ${XILINX_XCLBIN})
add_dependencies(${APP_EXECUTABLE} emconfig.json)
if (APP_DEPENDS)
add_dependencies(${APP_EXECUTABLE} ${APP_DEPENDS})
endif()
if (XILINX_LANGUAGE STREQUAL "opencl")
target_link_libraries(${APP_EXECUTABLE} rt pthread OpenCL)
else() # XILINX_LANGUAGE
target_link_libraries(${APP_EXECUTABLE} rt pthread)
endif() # XILINX_LANGUAGE
if (APP_LINK_LIB)
target_link_libraries(${APP_EXECUTABLE} ${APP_LINK_LIB})
endif() #APP_LINK_LIB
endif() #XILINX_OPENCL_SOURCES or XILINX_KERNEL_SOURCES
else() # APP_SOURCES AND USE_XILINX
if (APP_LIB_SOURCES AND USE_XILINX)
if (XILINX_OPENCL_SOURCES OR XILINX_KERNEL_SOURCES)
add_dependencies(${APP_LIBRARY_NAME} ${XILINX_XCLBIN})
add_dependencies(${APP_LIBRARY_NAME} emconfig.json)
if (XILINX_LANGUAGE STREQUAL "opencl")
target_link_libraries(${APP_LIBRARY_NAME} rt pthread OpenCL)
else() #XILINX_LANGUAGE
target_link_libraries(${APP_LIBRARY_NAME} rt pthread)
endif() #XILINX_LANGUAGE
endif() #XILINX_OPENCL_SOURCES XILINX_KERNEL_SOURCES
endif() #APP_LIB_SOURCES AND USE_XILINX
endif() #APP_SOURCES AND USE_XILINX

if (APP_SOURCES AND APP_LIBRARIES)
target_link_libraries(${APP_EXECUTABLE} ${APP_LIBRARIES})
endif()
if (APP_LIB_SOURCES AND APP_LIBRARIES)
target_link_libraries(${APP_LIBRARY_NAME} ${APP_LIBRARIES})
endif()
if (APP_SOURCES AND OPENMP_KERNEL_SOURCES)
target_link_libraries(${APP_EXECUTABLE} kernel.openmp)
add_dependencies(${APP_EXECUTABLE} kernel.openmp)
endif()

if (EXISTS "$ENV{IRIS}/lib64/libiris.so")
if (SNAPDRAGON)
install(FILES $ENV{IRIS}/lib64/libiris.so DESTINATION ${LIB_INSTALL_PATH})
else()
install(CODE "EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink $ENV{IRIS}/lib64/libiris.so ${LIB_INSTALL_PATH}/libiris.so)")
endif()
endif()

if (EXISTS "$ENV{IRIS}/lib/libiris.so")
if (SNAPDRAGON)
install(FILES $ENV{IRIS}/lib/libiris.so DESTINATION ${LIB_INSTALL_PATH})
else()
install(CODE "EXECUTE_PROCESS(COMMAND  ${CMAKE_COMMAND} -E create_symlink $ENV{IRIS}/lib/libiris.so ${LIB_INSTALL_PATH}/libiris.so)")
endif()
endif()

