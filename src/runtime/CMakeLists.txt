option(BUILD_SHARED_LIBS "Build shared libs" ON)

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

configure_file(Config.h.in Config.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable")
if (CMAKE_VERSION VERSION_LESS "3.1")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
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
if (TODO_DISABLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTODO_DISABLE")
endif()
if (CHECK_DISABLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCHECK_DISABLE")
endif()
if (WARNING_DISABLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWARNING_DISABLE")
endif()

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
  LoaderDefaultKernel.cpp
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
  LoaderDefaultKernel.h
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

if(USE_PYTHON)
  MESSAGE(STATUS "USE_PYTHON")
  install(FILES iris.py DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

add_library(iris ${IRIS_RUNTIME_SOURCES})
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

