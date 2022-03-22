cmake_minimum_required(VERSION 3.0)

project(IRIS)
include(GNUInstallDirs)

set(IRIS_VERSION_MAJOR 1)
set(IRIS_VERSION_MINOR 0)
set(IRIS_VERSION_PATCH 0)

set(IRIS_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(IRIS_INC_DIR ${PROJECT_SOURCE_DIR}/include)
set(IRIS_BIN_DIR ${PROJECT_BINARY_DIR}/src)

option(USE_PYTHON       "Enable support for Python"     ON )
option(USE_FORTRAN      "Enable support for Fortran 90" OFF)

option(USE_SIGHANDLER   "Enable signal handler"         OFF)
option(USE_HUB          "Enable hub"                    OFF)

add_subdirectory(include)
add_subdirectory(src)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/utils DESTINATION ${CMAKE_INSTALL_PREFIX}/.)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/setup.source
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "SCRIPT_FILE=\${BASH_SOURCE[0]}" > ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "SCRIPT=$(python -c \"import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))\" \${SCRIPT_FILE})" >> ${CMAKE_BINARY_DIR}/setup.source VERBATIM
  COMMAND echo "export IRIS=$(dirname -- \${SCRIPT})" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export LD_LIBRARY_PATH=.:\$IRIS/lib:\$IRIS/lib64:\$LD_LIBRARY_PATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "export PYTHONPATH=\$IRIS/include/iris:\$PYTHONPATH" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"*********************** IRIS Environent *********************\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"IRIS: \$IRIS\"" >> ${CMAKE_BINARY_DIR}/setup.source
  COMMAND echo "echo \"*************************************************************\"" >> ${CMAKE_BINARY_DIR}/setup.source
  )
add_custom_target(setup.source ALL DEPENDS ${CMAKE_BINARY_DIR}/setup.source)
install(FILES ${CMAKE_BINARY_DIR}/setup.source DESTINATION ${CMAKE_INSTALL_PREFIX})
