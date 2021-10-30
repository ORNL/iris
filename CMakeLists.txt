cmake_minimum_required(VERSION 2.8)

project(BRISBANE)
include(GNUInstallDirs)

set(BRISBANE_VERSION_MAJOR 1)
set(BRISBANE_VERSION_MINOR 0)
set(BRISBANE_VERSION_PATCH 0)

set(BRISBANE_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(BRISBANE_INC_DIR ${PROJECT_SOURCE_DIR}/include)
set(BRISBANE_BIN_DIR ${PROJECT_BINARY_DIR}/src)

set(USE_PYTHON  OFF)
set(USE_FORTRAN OFF)
set(USE_SIGHANDLER  OFF)
set(USE_HUB "Enable hub" OFF)

add_subdirectory(include)
add_subdirectory(src)
