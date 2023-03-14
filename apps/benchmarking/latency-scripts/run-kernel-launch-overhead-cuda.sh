#!/bin/bash
source ./setup.sh

export WORKING_DIR=`pwd`

make clean
make kernellaunch-cuda-profiling kernellaunch-iris-profiling kernel.ptx

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f kernellaunch-cuda-profiling ] || ! [ -f kernellaunch-iris-profiling ] || ! [ -f kernel.ptx ] ; then
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
./kernellaunch-cuda-profiling 1 1 1000    kernellaunch-cuda-${MACHINE}-1.csv
./kernellaunch-cuda-profiling 1 10 1000   kernellaunch-cuda-${MACHINE}-10.csv
./kernellaunch-cuda-profiling 1 100 1000  kernellaunch-cuda-${MACHINE}-100.csv
./kernellaunch-cuda-profiling 1 1000 1000 kernellaunch-cuda-${MACHINE}-1000.csv

#run IRIS single device
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 1 1000    kernellaunch-iris-cuda-${MACHINE}-1.csv
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 10 1000   kernellaunch-iris-cuda-${MACHINE}-10.csv
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 100 1000  kernellaunch-iris-cuda-${MACHINE}-100.csv
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 1000 1000 kernellaunch-iris-cuda-${MACHINE}-1000.csv

#run IRIS multi-device
#multiple GPU requires pooling to be enabled in iris:
sed -i 's/#define BRISBANE_POOL_ENABLED     0/#define BRISBANE_POOL_ENABLED     1/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_TASK    1100/#define BRISBANE_POOL_MAX_TASK    9999/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_CMD     1100/#define BRISBANE_POOL_MAX_CMD     9999/g' ../../src/runtime/Pool.h

source ./setup.sh

IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 1 1000    kernellaunch-multigpu-iris-cuda-${MACHINE}-1.csv
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 10 1000   kernellaunch-multigpu-iris-cuda-${MACHINE}-10.csv
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 100 1000  kernellaunch-multigpu-iris-cuda-${MACHINE}-100.csv
IRIS_ARCHS=cuda ./kernellaunch-iris-profiling 1 1000 1000 kernellaunch-multigpu-iris-cuda-${MACHINE}-1000.csv

#and back to single-gpu iris
sed -i 's/#define BRISBANE_POOL_ENABLED     1/#define BRISBANE_POOL_ENABLED     0/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_TASK    9999/#define BRISBANE_POOL_MAX_TASK    1100/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_CMD     9999/#define BRISBANE_POOL_MAX_CMD     1100/g' ../../src/runtime/Pool.h

source ./setup.sh

