install(FILES
  default_cpu_gpu_kernels.cpp
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
  Tiling1D.h
  Tiling2D.h
  Tiling3D.h
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
  hexagon/stub.h
  hexagon/stub_imp.h
  hexagon/hvx_util.h
  hexagon/q6cache.h
  hexagon/std_kernels.h
  hexagon/iris_interface.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/hexagon)

install(FILES
  hip/hip_runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/hip)

install(FILES
  level_zero/ze_api.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris/level_zero)

