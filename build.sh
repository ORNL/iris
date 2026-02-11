#!/bin/bash

is_macos() {
    [[ "$(uname)" == "Darwin" ]]
}


if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.iris"
fi
if [ ! -n "$IRIS_TAG" ]; then
IRIS_TAG=
fi
if [ ! -n "$BUILD_DIR" ]; then
    BUILD_DIR=build
fi
BUILD_TYPE=${BUILD_TYPE:-RELEASE}
BUILD_DIR_NAME=${BUILD_DIR}${IRIS_TAG}
rm -rf $BUILD_DIR_NAME
mkdir -p $BUILD_DIR_NAME
cd $BUILD_DIR_NAME
CC=${CC:-gcc}
CXX=${CXX:-g++}
CXX_STD=${CXX_STD:-11}
IRIS_FLAGS=

# Default preference: Ninja generator
NINJA=${NINJA:--G Ninja}
NO_NINJA=${NO_NINJA:-0}

NINJA_GEN=${NINJA_GEN:--DCMAKE_GENERATOR:INTERNAL=Ninja}
# Disable Ninja explicitly
if [[ "x${NO_NINJA}" == "x1" ]]; then
  NINJA=""
  NINJA_GEN=""
fi

if [[ "$(uname)" == "Darwin" ]]; then
    CC="gcc-15"
    CXX="g++-15"
    IRIS_FLAGS="-I/opt/homebrew/opt/gcc/include/c++/15/ -I/opt/homebrew/opt/gcc/include/c++/15/aarch64-apple-darwin24/"
fi

# Auto-fallback if Ninja is not installed
if [[ "${NINJA}" == "-G Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "Ninja not found. Falling back to default CMake generator."
  NINJA=""
  NINJA_GEN=""
  NO_NINJA=1
fi
#FC=gfortran
cmake ${NINJA} .. \
  ${NINJA_GEN} \
  -DCMAKE_INSTALL_PREFIX=${IRIS_INSTALL_ROOT}${IRIS_TAG}  \
  -DCMAKE_C_COMPILER=$CC      \
  -DCMAKE_CXX_COMPILER=$CXX   \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_CXX_STANDARD=${CXX_STD} \
  -DCMAKE_CXX_FLAGS="${IRIS_FLAGS}" \
  -DENABLE_FFI=ON \
  -DTRACE_DISABLE=ON  \
  -DDEBUG_DISABLE=ON  \
  -DINFO_DISABLE=ON   \
  -DCHECK_DISABLE=ON  \
  -DTODO_DISABLE=ON   \
  -DWARNING_DISABLE=ON \
  -DAUTO_PARALLEL=OFF  \
  -DAUTO_FLUSH=OFF  $@ 
  #-DUSE_PYTHON=ON     \
  #-DUSE_FORTRAN=ON    \

#cmake .. -DCMAKE_INSTALL_PREFIX=${IRIS_INSTALL_ROOT} -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=RELWITHDEBINFO -DTRACE_DISABLE=ON -DDEBUG_DISABLE=ON
#cmake .. -DCMAKE_INSTALL_PREFIX=$IRIS_INSTALL_ROOT -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=RELEASE -DTRACE_DISABLE=ON
#cmake .. -DCMAKE_INSTALL_PREFIX=$IRIS_INSTALL_ROOT -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=RELEASE -DUSE_PYTHON=ON -DUSE_FORTRAN=ON
if [[ "x${NO_NINJA}" == "x0" ]]; then
ninja install
else
make -j install
fi
