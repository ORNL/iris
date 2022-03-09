install(FILES
  iris.h
  iris.hpp
  iris_errno.h
  iris_hexagon.h
  iris_hexagon_imp.h
  iris_llvm.h
  iris_openmp.h
  iris_poly.h
  iris_poly_types.h
  iris_runtime.h
  iris_host2opencl.h
  iris_host2cuda.h
  iris_host2hip.h
  gettime.h
  verify.h
  iris_macros.h
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
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)

install(FILES
  CL/cl.h
  CL/cl_platform.h
  CL/cl_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/CL)

install(FILES
  cuda/cuda.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/cuda)

install(FILES
  hexagon/rpcmem.h
  hexagon/AEEStdDef.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/hexagon)

install(FILES
  hip/hip_runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/hip)

install(FILES
  level_zero/ze_api.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/level_zero)

