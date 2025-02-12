add_subdirectory(runtime)

install(FILES
  IrisHRT.jl
  DESTINATION ${CMAKE_INSTALL_PREFIX}/src)

install(FILES
  IrisHRT.jl
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)

