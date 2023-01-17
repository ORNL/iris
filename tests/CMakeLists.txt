cmake_minimum_required(VERSION 3.0)

project(IRIS-Tests)
include(GNUInstallDirs)

find_package(CUDA)
message(STATUS "CUDA_TOOL_KIT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_NVCC_EXECUTABLE=${CUDA_NVCC_EXECUTABLE}")
message(STATUS "CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY}")
#message(STATUS "CUDA_CUBLAS_LIBRARIES=${CUDA_CUBLAS_LIBRARIES}")
enable_testing()

# Format list(APPEND TEST_EXES "directory" "test1::test2")
# Space is used for an empty test exe

list(APPEND TEST_EXES "01_init_finalize" "test01_init_finalize" "")
list(APPEND TEST_EXES "02_task_depend" "test02_task_depend" "")
list(APPEND TEST_EXES "03_release" "test03_release" "")
list(APPEND TEST_EXES "04_enclosing_targets" "test04_enclosing_targets" "")
#[DEBUG] Remove Snapdragon-specific test.
#list(APPEND TEST_EXES "05_snapdragon" "test05_snapdragon" "")
list(APPEND TEST_EXES "06_device" "test06_device" "")
list(APPEND TEST_EXES "07_policy_register" "test07_policy_register" "")
list(APPEND TEST_EXES "08_multithreading" "test08_multithreading" "")
list(APPEND TEST_EXES "09_dataflow" "test09_dataflow" "")
list(APPEND TEST_EXES "10_multikernelexecution" "test10_multikernelexecution" "")
list(APPEND TEST_EXES "11_deadlock" "test11_deadlock" "")
list(APPEND TEST_EXES "12_task_custom" "test12_task_custom" "")
list(APPEND TEST_EXES "13_hooks" "test13_hooks" "")
list(APPEND TEST_EXES "14_permanent_task" "test14_permanent_task" "")
list(APPEND TEST_EXES "15_graph" "test15_graph" "")
#[FIXME] Temporarily disabled due to a deadlock issue.
#list(APPEND TEST_EXES "16_task_host" "test16_task_host" "")
list(APPEND TEST_EXES "17_json" "test17_json" "")
list(APPEND TEST_EXES "18_record" "test18_record" "")
#[DEBUG] Remove test20 due to dependency to cublas.
#list(APPEND TEST_EXES "20_cublas" "test20_cublas" "")
list(APPEND TEST_EXES "21_task_malloc" "test21_task_malloc" "")
list(APPEND TEST_EXES "23_multigraph" "test23_multigraph" "")
list(APPEND TEST_EXES "24_multi_kernels" "test24_multi_kernels" "")
list(APPEND TEST_EXES "25_random_tasks" "test25_random_tasks" "")
list(APPEND TEST_EXES "26_env_set" "test26_env_set" "")
#[FIXME] Temporarily disabled due to a deadlock issue.
list(APPEND TEST_EXES "27_deadlock2" "test27_deadlock2" "")
list(APPEND TEST_EXES "28_json2" "test28_json2" "")
#[DEBUG] Remove this test since exclusive memory has been removed.
#list(APPEND TEST_EXES "29_exclusive_mem" "test29_exclusive_mem" "")
list(APPEND TEST_EXES "30_task_info" "test30_task_info" " ")
#[FIXME] Temporarily disabled due to improper configuration.
#list(APPEND TEST_EXES "31_isaxpy" "test31_isaxpy" "")
list(APPEND TEST_EXES "32_json3" "test32_json3" "--logfile=time.csv --repeats=1 --scheduling-policy=roundrobin --size=100  --kernels=bigk --duplicates=2 --buffers-per-kernel=\"bigk:w r r\" --kernel-dimensions=bigk:2 --kernel-split=100 --depth=3 --num-tasks=3 --min-width=1 --max-width=1 --sandwich")
list(APPEND TEST_EXES "33_graph_cpp" "test33_graph_cpp" " ")

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
      COMMAND make IRIS=${IRIS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
   )
   else()
   add_custom_target(
      ${dir}
      ALL
      COMMAND make
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
   )
   endif()

   string(REPLACE "::" ";" exe_list ${exes})
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
    set_property(TEST ${exe} PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64:$ENV{LD_LIBRARY_PATH};PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH}")
   ENDFOREACH()
ENDFOREACH()
