install(FILES
  brisbane.h
  brisbane.hpp
  brisbane_errno.h
  brisbane_hexagon.h
  brisbane_hexagon_imp.h
  brisbane_llvm.h
  brisbane_openmp.h
  brisbane_poly.h
  brisbane_poly_types.h
  brisbane_runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)

install(FILES
  CL/cl.h
  CL/cl_platform.h
  CL/cl_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane/CL)

install(FILES
  cuda/cuda.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane/cuda)

install(FILES
  hexagon/rpcmem.h
  hexagon/AEEStdDef.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane/hexagon)

install(FILES
  hip/hip_runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane/hip)

install(FILES
  level_zero/ze_api.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane/level_zero)

