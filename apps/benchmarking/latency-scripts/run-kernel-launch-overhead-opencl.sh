#!/bin/bash
source ./setup.sh

export WORKING_DIR=`pwd`

make clean
if [ "$SYSTEM" = "leconte" ] ; then
  echo "OpenCL is not supported on Leconte." && exit
fi

make kernellaunch-opencl-profiling kernellaunch-iris-profiling

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f kernellaunch-opencl-profiling ] || ! [ -f kernellaunch-iris-profiling ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

#<application name> <memory size> <number of kernels to queue> <number of statistical samples> <log file to store samples>
#run CUDA baseline
./kernellaunch-opencl-profiling 1 1 1000    kernellaunch-opencl-${MACHINE}-1.csv
./kernellaunch-opencl-profiling 1 10 1000   kernellaunch-opencl-${MACHINE}-10.csv
./kernellaunch-opencl-profiling 1 100 1000  kernellaunch-opencl-${MACHINE}-100.csv
./kernellaunch-opencl-profiling 1 1000 1000 kernellaunch-opencl-${MACHINE}-1000.csv

#run IRIS single device
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 1 1000    kernellaunch-iris-opencl-${MACHINE}-1.csv
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 10 1000   kernellaunch-iris-opencl-${MACHINE}-10.csv
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 100 1000  kernellaunch-iris-opencl-${MACHINE}-100.csv
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 1000 1000 kernellaunch-iris-opencl-${MACHINE}-1000.csv

#run IRIS multi-device
#multiple GPU requires pooling to be enabled in iris:
sed -i 's/#define BRISBANE_POOL_ENABLED     0/#define BRISBANE_POOL_ENABLED     1/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_TASK    1100/#define BRISBANE_POOL_MAX_TASK    9999/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_CMD     1100/#define BRISBANE_POOL_MAX_CMD     9999/g' ../../src/runtime/Pool.h

source ./setup.sh

IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 1 1000    kernellaunch-multigpu-iris-opencl-${MACHINE}-1.csv
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 10 1000   kernellaunch-multigpu-iris-opencl-${MACHINE}-10.csv
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 100 1000  kernellaunch-multigpu-iris-opencl-${MACHINE}-100.csv
IRIS_ARCHS=opencl ./kernellaunch-iris-profiling 1 1000 1000 kernellaunch-multigpu-iris-opencl-${MACHINE}-1000.csv

#and back to single-gpu iris
sed -i 's/#define BRISBANE_POOL_ENABLED     1/#define BRISBANE_POOL_ENABLED     0/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_TASK    9999/#define BRISBANE_POOL_MAX_TASK    1100/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_CMD     9999/#define BRISBANE_POOL_MAX_CMD     1100/g' ../../src/runtime/Pool.h

source ./setup.sh

##!/bin/bash
#
##load modules and use latest gcc
#source /etc/profile.d/z00_lmod.sh
#module load gnu/8.3.0
##module load gnu/10.2.0
#export CC=gcc
#export CXX=g++
#
#MACHINE="Unknown"
#if [ `hostname` == "oswald00.ftpn.ornl.gov" ]
#then
#    export OPENCL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/cuda/11.4
#    MACHINE="Oswald"
#elif [ `hostname` == "equinox.ftpn.ornl.gov" ]
#then
#    export OPENCL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2021/cuda/11.0
#    MACHINE="Equinox"
#elif [ `hostname` == "radeon" ]
#then
#    export OPENCL_PATH=/opt/rocm/opencl
#    MACHINE="Radeon"
#elif [ `hostname` == "explorer" ]
#then
#    export OPENCL_PATH=/opt/rocm/opencl
#    MACHINE="Explorer"
#fi
#
#export OPENCL_CFLAGS="-I$OPENCL_PATH/include"
#export OPENCL_LDFLAGS="-L$OPENCL_PATH/lib64 -L$OPENCL_PATH/lib -Wl,-rpath=$OPENCL_PATH/lib64  -lOpenCL"
#
#export CXX_FLAGS+="-I$HOME/.local/include"
#export LD_FLAGS+="-L$HOME/.local/lib64 -lbrisbane -lrt $OPENCL_LDFLAGS"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.local/lib64"
#
##now opencl evaluation on leconte
#rm -f kernellaunch-opencl-${MACHINE}-*.csv kernellaunch-iris-opencl-${MACHINE}-*.csv kernellaunch-multigpu-iris-opencl-${MACHINE}-*.csv
#make clean
#make kernellaunch-opencl-profiling
#
##run OpenCL baseline
#./kernellaunch-opencl-profiling 1 1 1000    kernellaunch-opencl-${MACHINE}-1.csv
#./kernellaunch-opencl-profiling 1 10 1000   kernellaunch-opencl-${MACHINE}-10.csv
#./kernellaunch-opencl-profiling 1 100 1000  kernellaunch-opencl-${MACHINE}-100.csv
#./kernellaunch-opencl-profiling 1 1000 1000 kernellaunch-opencl-${MACHINE}-1000.csv
#
##rebuild iris to favour OpenCL for this test
#sed -i 's/openmp:cuda:hip:levelzero:hexagon:opencl/opencl:openmp:cuda:hip:levelzero:hexagon/g' ../../src/runtime/Platform.cpp
#cd ../.. ; ./build.sh ; cd apps/sc20
#make kernellaunch-iris-profiling
#
##run IRIS single device
#./kernellaunch-iris-profiling 1 1 1000    kernellaunch-iris-opencl-${MACHINE}-1.csv
#./kernellaunch-iris-profiling 1 10 1000   kernellaunch-iris-opencl-${MACHINE}-10.csv
#./kernellaunch-iris-profiling 1 100 1000  kernellaunch-iris-opencl-${MACHINE}-100.csv
#./kernellaunch-iris-profiling 1 1000 1000 kernellaunch-iris-opencl-${MACHINE}-1000.csv
#
##run IRIS multi-device
##multiple GPU requires pooling to be enabled in iris:
#sed -i 's/#define BRISBANE_POOL_ENABLED     0/#define BRISBANE_POOL_ENABLED     1/g' ../../src/runtime/Pool.h
#sed -i 's/#define BRISBANE_POOL_MAX_TASK    1100/#define BRISBANE_POOL_MAX_TASK    9999/g' ../../src/runtime/Pool.h
#sed -i 's/#define BRISBANE_POOL_MAX_CMD     1100/#define BRISBANE_POOL_MAX_CMD     9999/g' ../../src/runtime/Pool.h
#
#cd ../.. ; ./build.sh ; cd apps/sc20
#./kernellaunch-iris-profiling 1 1 1000    kernellaunch-multigpu-iris-opencl-${MACHINE}-1.csv
#./kernellaunch-iris-profiling 1 10 1000   kernellaunch-multigpu-iris-opencl-${MACHINE}-10.csv
#./kernellaunch-iris-profiling 1 100 1000  kernellaunch-multigpu-iris-opencl-${MACHINE}-100.csv
#./kernellaunch-iris-profiling 1 1000 1000 kernellaunch-multigpu-iris-opencl-${MACHINE}-1000.csv
#
##and back to single-gpu iris with default device preference
#sed -i 's/#define BRISBANE_POOL_ENABLED     1/#define BRISBANE_POOL_ENABLED     0/g' ../../src/runtime/Pool.h
#sed -i 's/#define BRISBANE_POOL_MAX_TASK    9999/#define BRISBANE_POOL_MAX_TASK    1100/g' ../../src/runtime/Pool.h
#sed -i 's/#define BRISBANE_POOL_MAX_CMD     9999/#define BRISBANE_POOL_MAX_CMD     1100/g' ../../src/runtime/Pool.h
#sed -i 's/opencl:openmp:cuda:hip:levelzero:hexagon/openmp:cuda:hip:levelzero:hexagon:opencl/g' ../../src/runtime/Platform.cpp
#cd ../.. ; ./build.sh ; cd apps/sc20
#
