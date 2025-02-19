add_subdirectory(runtime)

install(FILES
  IrisHRT.jl
  Tiling2D.jl
  Tiling3D.jl
  DESTINATION ${CMAKE_INSTALL_PREFIX}/src)

install(FILES
  IrisHRT.jl
  Tiling2D.jl
  Tiling3D.jl
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)

