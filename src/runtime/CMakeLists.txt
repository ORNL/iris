option(BUILD_SHARED_LIBS "Build shared libs" ON)

message(STATUS "CXX STD: ${CMAKE_CXX_STANDARD}")
if (ENABLE_FFI) 
include(CPM.cmake)
if (CMAKE_C_COMPILER MATCHES "nvc$")
CPMAddPackage(NAME libffi
        GITHUB_REPOSITORY "miniskar/libffi-cmake"
        OPTIONS "HAVE_AS_X86_PCREL"
        GIT_TAG master)
else()
CPMAddPackage(NAME libffi
        GITHUB_REPOSITORY "miniskar/libffi-cmake"
        GIT_TAG master)
endif()
include_directories(${libffi_BINARY_DIR}/include)
#add_subdirectory(libffi)
endif()
include_directories(${IRIS_INC_DIR})
include_directories(${IRIS_BIN_DIR}/runtime)
include_directories(${IRIS_BIN_DIR}/runtime/rapidjson)

#cmake_minimum_required(VERSION 3.15)
project(SetupPythonVenv)

# Step 1: Locate Python
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Step 2: Define virtual environment path
set(VENV_DIR "iris.venv")
set(VENV_PYTHON "${CMAKE_BINARY_DIR}/${VENV_DIR}/bin/python3")

# Step 3: Check if venv already exists, otherwise create it

# Step 4: Check if 'regex' is installed in the venv
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import regex"
    RESULT_VARIABLE regex_import_result
    OUTPUT_QUIET
    ERROR_QUIET
)

message(STATUS "VENV_PYTHON: ${VENV_PYTHON}")

# Step 5: If not installed, use venv pip to install regex
if(NOT regex_import_result EQUAL 0)
    message(STATUS "'regex' not found in venv. Installing via pip...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m venv "${VENV_DIR}"
        RESULT_VARIABLE pip_result
    )
    execute_process(
        COMMAND ${VENV_PYTHON} -m pip install regex
    )
    set(IRIS_PYTHON "${VENV_PYTHON}")
else()
    set(IRIS_PYTHON "python3")
    message(STATUS "'regex' already present in virtual environment.")
endif()


configure_file(Config.h.in Config.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable")
if (CMAKE_VERSION VERSION_LESS "3.1")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
  endif()
endif()
if (COVERAGE)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

if (USE_RISCV) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_RISCV")
endif()

if (IRIS_ASYNC_STREAMING) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIRIS_ASYNC_STREAMING")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIRIS_ASYNC_STREAMING")

if (ENABLE_SINGLE_DEVICE_PER_CU) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_SINGLE_DEVICE_PER_CU")
endif()

if (DISABLE_DYNAMIC_LINKING) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_DYNAMIC_LINKING")
endif()

if (DISABLE_CONSISTENCY_CHECK) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_CONSISTENCY_CHECK")
endif()

if (ENABLE_FFI) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_FFI")
endif()

if (TRACE_DISABLE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRACE_DISABLE")
endif()
if (INFO_DISABLE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINFO_DISABLE")
endif()
if (DEBUG_DISABLE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_DISABLE")
endif()

function(initVar var_name default_value)
#    message(STATUS "args var:(${var_name}) default_value:(${default_value})")
    if (NOT ${var_name})
        #message(STATUS "var  (${var_name}) is not defined. Setting default: ${default_value}")
        set(${var_name} "${default_value}" PARENT_SCOPE)
    endif() #var_name
    #message(STATUS "var  (${var_name}) is set to ${default_value}")
endfunction() # initVar
initVar(APP_FLAGS "-D__nothing__ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include/")
#message(STATUS " APP_FLAGS: ${APP_FLAGS}")
message(STATUS " PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS " CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
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
        get_filename_component(SRC_DIR_PATH ${SRC_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        get_filename_component(SRC_FILE_PATH ${first} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        list(APPEND src_include_paths "-I${SRC_DIR_PATH}")
        add_custom_command(OUTPUT  ${CMAKE_BINARY_DIR}/${rule_name}.iris 
            COMMAND ${IRIS_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/extract_iris_task_calls.py -input ${SRC_FILE_PATH} -compile_options "${APP_FLAGS} ${CMAKE_CXX_FLAGS}" -output ${CMAKE_BINARY_DIR}/${rule_name}.iris -extract  -codegen_file_tag iris_default_math
            DEPENDS ${SRC_FILE_PATH}
        )
        #message(STATUS " RULE: ${rule_name}")
        add_custom_target(${rule_name} DEPENDS ${CMAKE_BINARY_DIR}/${rule_name}.iris)
        #message(STATUS "\n Processing file: ${first} in path:${SRC_PATH} rule:${rule_name}\n")
        list(APPEND src_rules "${rule_name}")
        list(APPEND src_iris_files "${CMAKE_BINARY_DIR}/${rule_name}.iris")
   endwhile()
   add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/runtime/iris/iris_default_math_boilerplate.h ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_cuda_kernels.cu ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_hip_kernels.hip.cpp ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_openmp_kernels.cpp
           COMMAND mkdir -p ${CMAKE_BINARY_DIR}/src/runtime/iris
           COMMAND ${IRIS_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/extract_iris_task_calls.py -input ${src_iris_files} -output ${CMAKE_BINARY_DIR}/src/runtime/iris/iris_default_math_boilerplate.h -generate $ENV{IRIS_NO_CODEGEN} -codegen_file_tag iris_default_math -signature_file iris_default_math.h -signature_file_include_headers "iris/iris.h,iris/Tiling1D.h,iris/Tiling2D.h" -signature_file_using_namespace "std,iris"
           DEPENDS ${src_rules}
           )
    add_custom_target(iris_default_math_boilerplate_h DEPENDS ${CMAKE_BINARY_DIR}/src/runtime/iris/iris_default_math_boilerplate.h)
    install(FILES ${CMAKE_BINARY_DIR}/src/runtime/iris/iris_default_math_boilerplate.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/)
endfunction()



set(IRIS_RUNTIME_SOURCES
#AutoDAG.cpp
  BaseMem.cpp
  CAPI.cpp
  IRIS.cpp
  Command.cpp
  Consistency.cpp
  Device.cpp
  DeviceCUDA.cpp
  DeviceHexagon.cpp
  DeviceHIP.cpp
  DeviceLevelZero.cpp
  DeviceOpenCL.cpp
  DeviceOpenMP.cpp
  DeviceQIREE.cpp
  FilterTaskSplit.cpp
  Graph.cpp
  History.cpp
  HubClient.cpp
  JSON.cpp
  Kernel.cpp
  Loader.cpp
  LoaderCUDA.cpp
  LoaderResetKernel.cpp
  LoaderHost2OpenCL.cpp
  LoaderHost2HIP.cpp
  LoaderHost2CUDA.cpp
  LoaderHexagon.cpp
  LoaderHIP.cpp
  LoaderLevelZero.cpp
  LoaderOpenCL.cpp
  LoaderOpenMP.cpp
  LoaderPolicy.cpp
  LoaderQIREE.cpp
  Message.cpp
  DataMemRegion.cpp
  DataMem.cpp
  EventExchange.cpp
  Mem.cpp
  MemRange.cpp
  TGPolicy.cpp
  Platform.cpp
  Policies.cpp
  Policy.cpp
  HostInterface.cpp
  PolicyBlockCycle.cpp
  PolicyFirstToFinish.cpp
  PolicyData.cpp
  PolicyDefault.cpp
  PolicyDepend.cpp
  PolicyDevice.cpp
  PolicyJulia.cpp
  PolicyShortestDeviceQueue.cpp
  PolicyProfile.cpp
  PolicyRandom.cpp
  PolicyRoundRobin.cpp
  Polyhedral.cpp
  Pool.cpp
  PresentTable.cpp
  Profiler.cpp
  ProfilerDOT.cpp
  ProfilerEventRecord.cpp
  ProfilerGoogleCharts.cpp
  QueueReady.cpp
  QueueTask.cpp
  Reduction.cpp
  Retainable.cpp
  Scheduler.cpp
  SchedulingHistory.cpp
  SigHandler.cpp
  Task.cpp
  Thread.cpp
  Timer.cpp
  Utils.cpp
  Worker.cpp)

set(IRIS_RUNTIME_HEADERS
# AutoDAG.h
  Command.h
  Consistency.h
  Debug.h
  Device.h
  DeviceCUDA.h
  DeviceHexagon.h
  DeviceHIP.h
  HostInterface.h
  DeviceLevelZero.h
  DeviceOpenCL.h
  DeviceOpenMP.h
  DeviceQIREE.h
  Filter.h
  FilterTaskSplit.h
  Graph.h
  History.h
  HubClient.h
  Hub.h
  jsmn.h
  JSON.h
  Kernel.h
  Loader.h
  LoaderCUDA.h
  LoaderResetKernel.h
  LoaderHexagon.h
  LoaderHIP.h
  LoaderOpenCL.h
  LoaderOpenMP.h
  LoaderPolicy.h
  LoaderHost2OpenCL.h
  LoaderLevelZero.h
  LoaderHost2HIP.h
  LoaderHost2CUDA.h
  LoaderQIREE.h
  AsyncData.h
  EventExchange.h
  Message.h
  DataMemRegion.h
  DataMem.h
  BaseMem.h
  Mem.h
  MemRange.h
  Platform.h
  ObjectTrack.h
  Policies.h
  Policy.h
  PolicyBlockCycle.h
  PolicyFirstToFinish.h
  PolicyShortestDeviceQueue.h
  PolicyData.h
  PolicyDefault.h
  PolicyDevice.h
  PolicyDepend.h
  PolicyJulia.h
  PolicyProfile.h
  PolicyRandom.h
  PolicyRoundRobin.h
  Polyhedral.h
  Pool.h
  TGPolicy.h
  PresentTable.h
  Profiler.h
  ProfilerDOT.h
  ProfilerEventRecord.h
  ProfilerGoogleCharts.h
  QueueReady.h
  QueueTask.h
  Reduction.h
  Retainable.h
  Scheduler.h
  SchedulingHistory.h
  SigHandler.h
  Structs.h
  Task.h
  Thread.h
  Timer.h
  Utils.h
  Worker.h)


if (AUTO_PARALLEL) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUTO_PAR")
list(APPEND IRIS_RUNTIME_SOURCES AutoDAG.cpp)
list(APPEND IRIS_RUNTIME_HEADERS AutoDAG.h)
if (AUTO_FLUSH) 
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUTO_FLUSH")
endif()

if (AUTO_SHADOW) 
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUTO_SHADOW")
endif()

if (IGNORE_MANUAL_DEP) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIGNORE_MANUAL")
endif()

if (DEP_SANITY_CHECK) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSANITY_CHECK")
endif()

endif()

if (ENABLE_PER_TASK_COLOR_DOT) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPER_TASK_COLOR")
endif()

if (ENABLE_PRINT_TASK_DEP) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRINT_TASK_DEP")
endif()


if(USE_FORTRAN)
  MESSAGE(STATUS "USE_FORTRAN")
  enable_language(Fortran)
  list(APPEND IRIS_RUNTIME_SOURCES FortranAPI-IRIS.f90)
  install(FILES ${IRIS_BIN_DIR}/runtime/iris.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
endif()

install(FILES
  default_cpu_gpu_kernels.cpp
  DESTINATION ${CMAKE_INSTALL_PREFIX}/default)

if(USE_PYTHON)
  MESSAGE(STATUS "USE_PYTHON")
  install(FILES iris.py DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

set(DEFAULT_KERNELS 
    default_iris_task_core.cpp
    default_iris_task.cpp 
    )
extractIRISAPIs(${DEFAULT_KERNELS})

#add_library(default_iris ${DEFAULT_KERNELS})
#add_dependencies(default_iris  iris_app_cpu_dsp_interface.h)
#target_compile_options(default_iris PRIVATE -Wall)a
#install(TARGETS default_iris DESTINATION ${CMAKE_INSTALL_PREFIX}/default)

install(FILES ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/iris/)
install(FILES ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_hip_kernels.hip.cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/default/)
install(FILES ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_cuda_kernels.cu DESTINATION ${CMAKE_INSTALL_PREFIX}/default/)
#install(FILES ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_openmp_kernels.cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/default/)

add_library(iris ${IRIS_RUNTIME_SOURCES} ${DEFAULT_KERNELS})
add_dependencies(iris iris_default_math_boilerplate_h)
if (ENABLE_FFI)
install(FILES ${libffi_BINARY_DIR}/libffi.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${libffi_BINARY_DIR}/libffi.a DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${libffi_BINARY_DIR}/include/fficonfig.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${libffi_BINARY_DIR}/include/ffi.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${libffi_BINARY_DIR}/include/ffitarget.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set_target_properties(iris PROPERTIES LINK_FLAGS "-L${libffi_BINARY_DIR} ")
target_link_libraries(iris ffi)
endif() #ENABLE_FFI

target_compile_options(iris PRIVATE -Wall)

install(TARGETS iris DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  ${IRIS_RUNTIME_HEADERS}
  ${IRIS_BIN_DIR}/runtime/Config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/rt)

message(STATUS "NVCC: ${CUDA_NVCC_EXECUTABLE}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CUDA_COMPILER)
else()
find_package(OpenMP REQUIRED)
endif()
message(STATUS "OpenMP_CXX_FOUND: ${OpenMP_CXX_FOUND}")
add_library(cpu_default default_cpu_gpu_kernels.cpp ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_openmp_kernels.cpp)
add_dependencies(cpu_default iris_default_math_boilerplate_h)
target_compile_options(cpu_default PRIVATE -Wall -fopenmp -DENABLE_IRIS_OPENMP_APIS)
if(APPLE)
    target_compile_options(cpu_default PRIVATE  ${CMAKE_CXX_FLAGS})
endif()
if (CMAKE_CUDA_COMPILER)
else()
target_link_libraries(cpu_default PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS cpu_default DESTINATION ${CMAKE_INSTALL_PREFIX}/default)

find_package(CUDAToolkit QUIET)

if (CUDAToolkit_FOUND)
    find_program(NVCC_EXECUTABLE
        NAMES nvcc
        HINTS ${CUDAToolkit_ROOT}/bin
              $ENV{CUDA_PATH}/bin
              $ENV{NVHPC_ROOT}/cuda/bin
              /usr/local/cuda/bin

    )
    if (NVCC_EXECUTABLE)
        message(STATUS "CUDA found!")
        find_path(
                CURAND_INCLUDE_DIR
                NAMES curand.h
                HINTS "${CUDA_TOOLKIT_ROOT_DIR}/include"
                      "$ENV{MATH_LIBS}/include"
                      "$ENV{NVIDIA_PATH}/math_libs/include"
                      "$ENV{NVHPC_ROOT}/math_libs/include"
                PATH_SUFFIXES ""        # you can strip off sub-folders
                REQUIRED
              )
        find_library(
              CURAND_LIBRARY
              NAMES curand curand_static   # try both the shared and static names
              HINTS
                "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                "${CUDA_TOOLKIT_ROOT_DIR}/lib"
                "$ENV{MATH_LIBS}/lib"
                "$ENV{MATH_LIBS}/lib64"
                "$ENV{NVIDIA_PATH}/math_libs/lib"
                "$ENV{NVIDIA_PATH}/math_libs/lib64"
                "$ENV{NVHPC_ROOT}/math_libs/lib"
                "$ENV{NVHPC_ROOT}/math_libs/lib64"
              PATH_SUFFIXES ""             # no extra suffixes
              REQUIRED
            )
        get_filename_component(
            CURAND_LIBDIR   # the variable to set
            ${CURAND_LIBRARY}  # the input full path
            DIRECTORY      # tell CMake you want the directory portion
          )
        message(STATUS "Found curand library at: ${CURAND_LIBDIR}")
        message(STATUS "CURAND_INCLUDE_DIR imported targets:  ${CURAND_INCLUDE_DIR}")
        #find_package(CUDA REQUIRED COMPONENTS curand
        #            HINTS $ENV{MATH_PATH}/lib64)
        set(MYCUDA_SRCS
               default_cpu_gpu_kernels.cpp 
                ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_cuda_kernels.cu 
        )
        #get_property(ALL_TARGETS GLOBAL PROPERTY TARGETS)
        #message(STATUS "All imported targets:\n  ${ALL_TARGETS}")

        get_target_property(CUDA_INCLUDE_DIRS CUDA::cudart
                          INTERFACE_INCLUDE_DIRECTORIES)
        #get_target_property(MATHLIB_INCLUDE_DIRS CUDA::curand
        #                  INTERFACE_INCLUDE_DIRECTORIES)
        set(include_paths "-I${CURAND_INCLUDE_DIR}")
        #while(MATHLIB_INCLUDE_DIRS)
        #    list(GET MATHLIB_INCLUDE_DIRS 0 first_core)
        #    list(REMOVE_AT MATHLIB_INCLUDE_DIRS 0)
        #    list(APPEND include_paths -I${first_core})
        #endwhile()
        list(APPEND include_paths "-I${CMAKE_BINARY_DIR}/src/runtime")
        add_custom_command(
            OUTPUT libcuda_default.so
            #COMMAND nvcc -x cu -g -Xcompiler -fPIC -DENABLE_IRIS_HOST2CUDA_APIS ${include_paths} -I. -I${IRIS_INC_DIR} -I${CUDA_INCLUDE_DIRS}/ ${CMAKE_CURRENT_SOURCE_DIR}/default_cpu_gpu_kernels.cpp -c -o default_cpu_gpu_kernels_cuda.cpp.o
            #COMMAND nvcc -x cu -g -Xcompiler -fPIC -I${IRIS_INC_DIR} ${include_paths} -I. ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_cuda_kernels.cu -c -o iris_default_math_cuda_kernels.cu.o
            #COMMAND nvcc -shared -g -Xcompiler -fPIC default_cpu_gpu_kernels_cuda.cpp.o iris_default_math_cuda_kernels.cu.o -o libcuda_default.so 
			COMMAND nvcc -shared -x cu -g -Xcompiler -fPIC -DENABLE_IRIS_HOST2CUDA_APIS ${include_paths} -I. -I${IRIS_INC_DIR} -I${CUDA_INCLUDE_DIRS}/ ${CMAKE_CURRENT_SOURCE_DIR}/default_cpu_gpu_kernels.cpp ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_cuda_kernels.cu -o libcuda_default.so 
			DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/default_cpu_gpu_kernels.cpp
        )
        add_custom_target(myapp_cuda DEPENDS libcuda_default.so)
        add_dependencies(myapp_cuda iris_default_math_boilerplate_h)
        install(FILES  ${CMAKE_BINARY_DIR}/src/runtime/libcuda_default.so  DESTINATION ${CMAKE_INSTALL_PREFIX}/default)
        add_dependencies(iris myapp_cuda)
    endif()
endif()

find_package(hip 
        HINTS
        "/opt/rocm"               # root of your ROCm install
        "/opt/rocm/hip"           # sometimes hip lives in a separate sub-folder
        "$ENV{ROCM_PATH}"
        PATH_SUFFIXES
        lib/cmake/hip             # where hip-config.cmake usually resides
        )
if (hip_FOUND)
	message(STATUS "HIP found!")
 	get_target_property(HIP_INCLUDE_DIRS hip::host
                     INTERFACE_INCLUDE_DIRECTORIES)
	set(include_paths "")
	add_custom_command(
		OUTPUT libhip_default.so
		COMMAND hipcc -g -fPIC ${include_paths} -I. -I${CMAKE_BINARY_DIR}/src/runtime -DENABLE_IRIS_HOST2HIP_APIS -I${IRIS_INC_DIR} -I${HIP_INCLUDE_DIRS}/ ${CMAKE_CURRENT_SOURCE_DIR}/default_cpu_gpu_kernels.cpp -c -o default_cpu_gpu_kernels_hip.cpp.o
		COMMAND hipcc -g -fPIC -I${IRIS_INC_DIR} -I. -I${CMAKE_BINARY_DIR}/src/runtime ${CMAKE_BINARY_DIR}/src/runtime/iris_default_math_hip_kernels.hip.cpp -c -o iris_default_math_hip_kernels.hip.cpp.o
		COMMAND hipcc -shared -g -fPIC default_cpu_gpu_kernels_hip.cpp.o iris_default_math_hip_kernels.hip.cpp.o -o libhip_default.so 
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/default_cpu_gpu_kernels.cpp
	)
	add_custom_target(myapp_hip DEPENDS libhip_default.so)
	add_dependencies(myapp_hip iris_default_math_boilerplate_h)
	install(FILES  ${CMAKE_BINARY_DIR}/src/runtime/libhip_default.so  DESTINATION ${CMAKE_INSTALL_PREFIX}/default)
	add_dependencies(iris myapp_hip)
endif()
