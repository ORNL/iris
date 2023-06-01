option(BUILD_SHARED_LIBS "Build shared libs" ON)

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

if (IRIS_ASYNC_STREAMING) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIRIS_ASYNC_STREAMING")
endif()

if (ENABLE_SINGLE_DEVICE_PER_CU) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_SINGLE_DEVICE_PER_CU")
endif()

if (DISABLE_CONSISTENCY_CHECK) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_CONSISTENCY_CHECK")
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

set(IRIS_RUNTIME_SOURCES
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
  FilterTaskSplit.cpp
  Graph.cpp
  History.cpp
  HubClient.cpp
  JSON.cpp
  Kernel.cpp
  Loader.cpp
  LoaderCUDA.cpp
  LoaderHost2OpenCL.cpp
  LoaderHost2HIP.cpp
  LoaderHost2CUDA.cpp
  LoaderHexagon.cpp
  LoaderHIP.cpp
  LoaderLevelZero.cpp
  LoaderOpenCL.cpp
  LoaderOpenMP.cpp
  LoaderPolicy.cpp
  Message.cpp
  DataMemRegion.cpp
  DataMem.cpp
  Mem.cpp
  MemRange.cpp
  Platform.cpp
  Policies.cpp
  Policy.cpp
  PolicyAll.cpp
  PolicyAny.cpp
  PolicyData.cpp
  PolicyDefault.cpp
  PolicyDepend.cpp
  PolicyDevice.cpp
  PolicyProfile.cpp
  PolicyRandom.cpp
  PolicyRoundRobin.cpp
  Polyhedral.cpp
  Pool.cpp
  PresentTable.cpp
  Profiler.cpp
  ProfilerDOT.cpp
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
  Command.h
  Consistency.h
  Debug.h
  Device.h
  DeviceCUDA.h
  DeviceHexagon.h
  DeviceHIP.h
  DeviceLevelZero.h
  DeviceOpenCL.h
  DeviceOpenMP.h
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
  LoaderHost2OpenCL.h
  LoaderLevelZero.h
  LoaderHost2HIP.h
  LoaderHost2CUDA.h
  LoaderCUDA.h
  LoaderHexagon.h
  LoaderHIP.h
  LoaderOpenCL.h
  LoaderOpenMP.h
  LoaderPolicy.h
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
  PolicyAll.h
  PolicyAny.h
  PolicyData.h
  PolicyDefault.h
  PolicyDevice.h
  PolicyDepend.h
  PolicyProfile.h
  PolicyRandom.h
  PolicyRoundRobin.h
  Polyhedral.h
  Pool.h
  PresentTable.h
  Profiler.h
  ProfilerDOT.h
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

if(USE_FORTRAN)
  MESSAGE(STATUS "USE_FORTRAN")
  enable_language(Fortran)
  list(APPEND IRIS_RUNTIME_SOURCES FortranAPI-IRIS.f90)
  install(FILES ${IRIS_BIN_DIR}/runtime/iris.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
endif()

if(USE_PYTHON)
  MESSAGE(STATUS "USE_PYTHON")
  install(FILES iris.py DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
endif()

add_library(iris ${IRIS_RUNTIME_SOURCES})

target_compile_options(iris PRIVATE -Wall)

install(TARGETS iris DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  ${IRIS_RUNTIME_HEADERS}
  ${IRIS_BIN_DIR}/runtime/Config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/rt)

