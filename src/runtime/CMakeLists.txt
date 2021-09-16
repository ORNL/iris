option(BUILD_SHARED_LIBS "Build shared libs" ON)

include_directories(${BRISBANE_INC_DIR})
include_directories(${BRISBANE_BIN_DIR}/runtime)

configure_file(Config.h.in Config.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable")
if (CMAKE_VERSION VERSION_LESS "3.1")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

set(BRISBANE_RUNTIME_SOURCES
  CAPI.cpp
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
  LoaderHexagon.cpp
  LoaderHIP.cpp
  LoaderLevelZero.cpp
  LoaderOpenCL.cpp
  LoaderOpenMP.cpp
  LoaderPolicy.cpp
  Message.cpp
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
  SigHandler.cpp
  Task.cpp
  Thread.cpp
  Timer.cpp
  Utils.cpp
  Worker.cpp)

set(BRISBANE_RUNTIME_HEADERS
  Command.h
  Consistency.h
  Debug.h
  Device.h
  DeviceCUDA.h
  DeviceHexagon.h
  DeviceHIP.h
  DeviceOpenCL.h
  DeviceOpenMP.h
  FilterTaskSplit.h
  Graph.h
  History.h
  HubClient.h
  JSON.h
  Kernel.h
  Loader.h
  LoaderCUDA.h
  LoaderHexagon.h
  LoaderHIP.h
  LoaderOpenCL.h
  LoaderOpenMP.h
  LoaderPolicy.h
  Message.h
  Mem.h
  MemRange.h
  Platform.h
  Policies.h
  Policy.h
  PolicyAll.h
  PolicyAny.h
  PolicyData.h
  PolicyDefault.h
  PolicyDevice.h
  PolicyProfile.h
  PolicyRandom.h
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
  list(APPEND BRISBANE_RUNTIME_SOURCES FortranAPI-IRIS.f90)
  install(FILES ${BRISBANE_BIN_DIR}/runtime/iris.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
endif()

if(USE_PYTHON)
  MESSAGE(STATUS "USE_PYTHON")
  install(FILES iris.py DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
endif()

add_library(iris ${BRISBANE_RUNTIME_SOURCES})

target_compile_options(iris PRIVATE -Wall)

install(TARGETS iris DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  ${BRISBANE_RUNTIME_HEADERS}
  ${BRISBANE_BIN_DIR}/runtime/Config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/rt)

