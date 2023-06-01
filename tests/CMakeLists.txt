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

list(APPEND TEST_EXES "01_init_finalize" "test01_init_finalize" "")
if (CUDAToolkit_FOUND OR HIP_ROOT_DIR)
list(APPEND TEST_EXES "02_task_depend" "test02_task_depend" "")
endif()
list(APPEND TEST_EXES "03_release" "test03_release" "")
if (CUDAToolkit_FOUND OR HIP_ROOT_DIR)
list(APPEND TEST_EXES "04_enclosing_targets" "test04_enclosing_targets" "")
endif()
#[DEBUG] Remove Snapdragon-specific test.
#list(APPEND TEST_EXES "05_snapdragon" "test05_snapdragon" "")
list(APPEND TEST_EXES "06_device" "test06_device" "")
list(APPEND TEST_EXES "07_policy_register" "test07_policy_register" "")
list(APPEND TEST_EXES "08_multithreading" "test08_multithreading" "")
list(APPEND TEST_EXES "09_dataflow" "test09_dataflow" "")
if (CUDAToolkit_FOUND)
list(APPEND TEST_EXES "10_multikernelexecution" "test10_multikernelexecution" "")
endif()
#list(APPEND TEST_EXES "11_deadlock" "test11_deadlock" "")
#[REMOVE] CUDA specific and interfaces directly with the CUDA loader
#list(APPEND TEST_EXES "12_task_custom" "test12_task_custom" "")
list(APPEND TEST_EXES "13_hooks" "test13_hooks" "")
if (CUDAToolkit_FOUND)
list(APPEND TEST_EXES "14_permanent_task" "test14_permanent_task" "")
endif()
list(APPEND TEST_EXES "15_graph" "test15_graph" "")
#[FIXME] Temporarily disabled due to a deadlock issue.
#list(APPEND TEST_EXES "16_task_host" "test16_task_host" "")
list(APPEND TEST_EXES "17_json" "test17_json" "")
list(APPEND TEST_EXES "18_record" "test18_record" "")
list(APPEND TEST_EXES "19_replay" "test19_replay" "")
#[DEBUG] Remove test20 due to dependency to cublas.
if (CUDAToolkit_FOUND)
  #list(APPEND TEST_EXES "20_cublas" "test20_cublas" "")
list(APPEND TEST_EXES "21_task_malloc" "test21_task_malloc" "")
endif()
list(APPEND TEST_EXES "23_multigraph" "test23_multigraph" "")
list(APPEND TEST_EXES "24_multi_kernels" "test24_multi_kernels" "")
if (CUDAToolkit_FOUND)
list(APPEND TEST_EXES "25_random_tasks" "test25_random_tasks" "")
endif()
list(APPEND TEST_EXES "26_env_set" "test26_env_set" "")
list(APPEND TEST_EXES "27_deadlock2" "test27_deadlock2" "")
list(APPEND TEST_EXES "28_json2" "test28_json2" "")
list(APPEND TEST_EXES "29_data_mem" "test29_data_mem" "")
if (CUDAToolkit_FOUND)
list(APPEND TEST_EXES "30_task_info" "test30_task_info" " ")
endif()
#[FIXME] Temporarily disabled due to improper configuration.
#list(APPEND TEST_EXES "31_isaxpy" "test31_isaxpy" "")
list(APPEND TEST_EXES "32_json3" "test32_json3" "--kernels=ijk --duplicates=0 --buffers-per-kernel=\"ijk:w-r-r\" --kernel-dimensions=ijk:2 --kernel-split=100 --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --logfile=log.txt --repeats=1 --scheduling-policy=any --size=128")
list(APPEND TEST_EXES "33_graph_cpp" "test33_graph_cpp" " ")
list(APPEND TEST_EXES "34_set_mem" "test34_set_mem" " ")

# Run each test in TEST_EXES
list(LENGTH TEST_EXES count)
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
   FOREACH(exe ${exe_list})
      if(${exe} STREQUAL " ") # Space is used for an empty test.
         continue()
      endif()
      message(STATUS "**Test Added** Dir: ${dir} Exe: ${exe} Arg: ${args}")
      add_test(
         NAME ${exe}
         COMMAND ${exe} ${args}
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
         )
      if (CUDAToolkit_FOUND OR HIP_ROOT_DIR)
        set_property(TEST ${exe} PROPERTY ENVIRONMENT ${TEST_PROP})
      endif()
      #some tests use .so resources that are local (in the same directory), i.e. 07_policy_register, so add it to the LD_LIBRARY_PATH
      set_property(TEST ${exe} PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
   ENDFOREACH()
ENDFOREACH()
