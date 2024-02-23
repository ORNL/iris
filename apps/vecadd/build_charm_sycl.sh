#!/bin/bash

# You may need to point to path to where llvm-config lives for instance, 
# on your system:
#     export PATH=/usr/lib/llvm-14/bin:$PATH
#
# ideally using modules are supported:
#     module load llvm/13.0.1
#
# Or in the worst case using a prebuilt binary, i.e.:
#     wget https://github.com/awakecoding/llvm-prebuilt/releases/download/v2023.1.0/clang+llvm-14.0.6-x86_64-ubuntu-22.04.tar.xz
#     tar -xvf clang+llvm-14.0.6-x86_64-ubuntu-22.04.tar.xz
#     export PATH=`pwd`/clang+llvm-14.0.6-x86_64-ubuntu-22.04/bin:$PATH

#zenith
module load nvhpc/22.11
eval `spack env activate --sh charmsycl`
#module load llvm/14.0.0
export LLVM_DIR=/home/9bj/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-12.2.0/llvm-16.0.2-xtdhivs4mlbtj4wrj34hef3nqnjrxj5j

#oswald03
#module load llvm/13.0.1 cmake/3.26.3 nvhpc/23.3
export CC=$LLVM_DIR/bin/clang; export CXX=$LLVM_DIR/bin/clang++
export CMAKE_SYSTEM_PREFIX_PATH=$LLVM_DIR:$CMAKE_SYSTEM_PREFIX_PATH

git submodule update --init --recursive
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.iris"
fi
rm -rf build; mkdir build; cd build
cmake ..  -G "Ninja" -DCMAKE_GENERATOR:INTERNAL=Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_IRIS=TRUE -DCHARM_SYCL_IRIS_IS_REQUIRED=TRUE -DIRIS_DIR=$IRIS_INSTALL_ROOT -DCMAKE_INSTALL_PREFIX=~/.charm-sycl -DCMAKE_CXX_FLAGS="-std=c++14" -DCHARM_SYCL_USE_CLANG_DYLIB=NO -DCMAKE_POSITION_INDEPENDENT_CODE=YES
ninja
[ $? -ne 0 ] && exit
ninja install
[ $? -ne 0 ] && exit

#ninja check
#[ $? -ne 0 ] && exit

exit 0
#cmake ..  -DCMAKE_BUILD_TYPE=Release -DUSE_IRIS=TRUE -DCHARM_SYCL_IRIS_IS_REQUIRED=TRUE -DIRIS_DIR=$IRIS_INSTALL_ROOT -DCHARM_SYCL_ENABLE_ASAN=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=OFF -DBUILD_TESTING=TRUE -DCMAKE_INSTALL_PREFIX=~/.charm-sycl -DCMAKE_CXX_FLAGS="-std=c++14"
#make -j$(nproc)
#[ $? -ne 0 ] && exit
#make install
#[ $? -ne 0 ] && exit
#cd build; make check
#[ $? -ne 0 ] && exit
