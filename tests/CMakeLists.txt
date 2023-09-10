cmake_minimum_required(VERSION 3.0)

project(IRIS-Tests)
include(GNUInstallDirs)

find_package(CUDAToolkit)
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
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(CUDAToolkit_TARGET_NAME "x86_64-linux")
endif()

message(STATUS "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_BIN_DIR=${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDAToolkit_TARGET_NAME=${CUDAToolkit_TARGET_NAME}")
message(STATUS "CUDAToolkit_TARGET_DIR=${CUDAToolkit_TARGET_DIR}")

set(LD_LIBRARY_PATH "$ENV{LD_LIBRARY_PATH}")
set(SYS_PATH "$ENV{PATH}")
set(EXT_INCLUDE )
set(EXT_LDFLAGS )
if (CUDAToolkit_FOUND)
set(CUDAToolkit_ROOT_DIR ${CUDAToolkit_TARGET_DIR})
set(CUSTOM_CUDAToolkit_TARGET_DIR "${CUDAToolkit_ROOT_DIR}/targets/${CUDAToolkit_TARGET_NAME}")
set(CUDAToolkit_MATH_LIB_DIR "${CUDAToolkit_TARGET_DIR}/../../math_libs")
message(STATUS "CUSTOM_CUDAToolkit_TARGET_DIR=${CUSTOM_CUDAToolkit_TARGET_DIR}")
message(STATUS "CUDAToolkit_MATH_LIB_DIR=${CUDAToolkit_MATH_LIB_DIR}")
set(LD_LIBRARY_PATH "${CUDAToolkit_LIBRARY_DIR}/:${LD_LIBRARY_PATH}")
set(SYS_PATH "${CUDAToolkit_BIN_DIR}/.:${SYS_PATH}")
set(EXT_INCLUDE -I${CUDAToolkit_INCLUDE_DIRS} -I${CUSTOM_CUDAToolkit_TARGET_DIR}/include)
set(EXT_LDFLAGS -L${CUDAToolkit_LIBRARY_DIR} -L${CUDAToolkit_MATH_LIB_DIR}/lib64 -L${CUDAToolkit_MATH_LIB_DIR}/lib)
message(STATUS "CUDA Toolkit found")
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
find_package(HIP)
if (HIP_ROOT_DIR)
set(LD_LIBRARY_PATH "${HIP_ROOT_DIR}/lib64:${HIP_ROOT_DIR}/lib:${LD_LIBRARY_PATH}")
set(SYS_PATH "${HIP_ROOT_DIR}/bin:${SYS_PATH}")
else()
message(STATUS "HIP not found")
endif()
message(STATUS "HIP_ROOT_DIR=${HIP_ROOT_DIR}")
set(TEST_PROP "LD_LIBRARY_PATH=${LD_LIBRARY_PATH};PATH=${SYS_PATH}")

enable_testing()
# Format list(APPEND TEST_EXES "directory" "test1::test2")
# Space is used for an empty test exe

list(APPEND TEST_EXES "04_enclosing_targets" "test04_enclosing_targets" "")

# Run each test in TEST_EXES
list(LENGTH TEST_EXES count)
math(EXPR TEST_REPEAT 1)
math(EXPR count "${count}-1")
foreach(i RANGE 0 ${count} 3)
  math(EXPR ii "${i}+1")
  math(EXPR iii "${i}+2")

  list(GET TEST_EXES ${i} dir)
  list(GET TEST_EXES ${ii} exes)
  list(GET TEST_EXES ${iii} args)
  separate_arguments(args)

   message(STATUS "**Build Target Added** Dir: ${dir}")
   if (IRIS)
   add_custom_target(
      ${dir}
      ALL
      COMMAND make IRIS=${IRIS} CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} EXT_INCLUDE="${EXT_INCLUDE}" EXT_LDFLAGS="${EXT_LDFLAGS}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
   )
   else()
   add_custom_target(
      ${dir}
      ALL
      COMMAND make CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} EXT_INCLUDE="${EXT_INCLUDE}" EXT_LDFLAGS="${EXT_LDFLAGS}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
   )
   endif()

   string(REPLACE "::" ";" exe_list ${exes})
   message(STATUS "**LD_LIBRARY_PATH** : ${LD_LIBRARY_PATH}")
   FOREACH(index RANGE 1 ${TEST_REPEAT} 1)
      FOREACH(exe ${exe_list})
         if(${exe} STREQUAL " ") # Space is used for an empty test.
            continue()
         endif()
         message(STATUS "**Test Added** Dir: ${dir} Exe: ${exe} Iteration:${index} Arg: ${args}")
         set(exe_it "${exe}_i${index}")
         add_test(
            NAME ${exe_it}
            COMMAND bash -c "valgrind --leak-check=summary ./${exe} ${args}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
            )
         if (CUDAToolkit_FOUND OR HIP_ROOT_DIR)
           set_property(TEST ${exe_it} PROPERTY ENVIRONMENT ${TEST_PROP})
         endif()
         #some tests use .so resources that are local (in the same directory), i.e. 07_policy_register, so add it to the LD_LIBRARY_PATH
         set_property(TEST ${exe_it} PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
      ENDFOREACH()
   ENDFOREACH()
ENDFOREACH()
