cmake_minimum_required(VERSION 3.5)

project(ISaxpy)
set(XILINX_VERSION 1)
#set(XILINX_TARGET  "$ENV{XCL_EMULATION_MODE}")  # sw_emu, hw_emu, hw
set(XILINX_LANGUAGE "xilinx") # opencl, xilinx

set(XILINX_OPENCL_SOURCES 
   )
set(XILINX_KERNEL_SOURCES 
    saxpy              kernel_stage  src/kernel.xilinx.cpp
   )

#set(EXTRACT_IRIS_KERNEL_SIGNATURES
#    src/signature.def)

set(OPENMP_KERNEL_SOURCES
   src/kernel.cl.openmp.c
   )

set(HIP_KERNEL_SOURCES
   src/kernel.hip
   )

set(CUDA_KERNEL_SOURCES
   src/kernel.cu
   )

set(APP_SOURCES
   src/saxpy.cpp
   src/saxpy.iris.cpp
   src/saxpy_ref.cpp
   )

set(HEXAGON_KERNEL_SOURCES 
   src/kernel.cl.hexagon.c
   )

set(APP_EXECUTABLE  isaxpy)
set(OPENMP_FLAGS "-I$ENV{IRIS}/include/iris/hexagon")
include($ENV{IRIS}/utils/CMakeLists.txt)
