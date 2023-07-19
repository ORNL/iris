#!/bin/bash

source ./setup.sh
#ensure libiris.so is in the shared library path
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.iris"
fi
echo "ADDING $IRIS_INSTALL_ROOT/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$IRIS_INSTALL_ROOT/lib64:$IRIS_INSTALL_ROOT/lib:$LD_LIBRARY_PATH
export WORKING_DIR=`pwd`

make clean
make task-latency-cuda task-latency-iris kernel.ptx

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f task-latency-cuda ] || ! [ -f task-latency-iris ] || ! [ -f kernel.ptx ] ; then
  exit
fi

#<application name> <memory size> <number of kernels to queue> <number of statistical samples> <log file to store samples>
#run CUDA baseline
./task-latency-cuda 1 1 1000    kernellaunch-cuda-${HOST}-1.csv
./task-latency-cuda 1 10 1000   kernellaunch-cuda-${HOST}-10.csv
./task-latency-cuda 1 100 1000  kernellaunch-cuda-${HOST}-100.csv
./task-latency-cuda 1 1000 1000 kernellaunch-cuda-${HOST}-1000.csv

#run IRIS single device
IRIS_ARCHS=cuda ./task-latency-iris 1 1 1000    kernellaunch-iris-cuda-${HOST}-1.csv
IRIS_ARCHS=cuda ./task-latency-iris 1 10 1000   kernellaunch-iris-cuda-${HOST}-10.csv
IRIS_ARCHS=cuda ./task-latency-iris 1 100 1000  kernellaunch-iris-cuda-${HOST}-100.csv
IRIS_ARCHS=cuda ./task-latency-iris 1 1000 1000 kernellaunch-iris-cuda-${HOST}-1000.csv

#run IRIS multi-device
#multiple GPU requires pooling to be enabled in iris:
sed -i 's/#define IRIS_POOL_ENABLED     0/#define IRIS_POOL_ENABLED     1/g' ../../src/runtime/Pool.h
sed -i 's/#define IRIS_POOL_MAX_TASK    1100/#define IRIS_POOL_MAX_TASK    9999/g' ../../src/runtime/Pool.h
sed -i 's/#define IRIS_POOL_MAX_CMD     1100/#define IRIS_POOL_MAX_CMD     9999/g' ../../src/runtime/Pool.h

source ./setup.sh

IRIS_ARCHS=cuda ./task-latency-iris 1 1 1000    kernellaunch-multigpu-iris-cuda-${HOST}-1.csv
IRIS_ARCHS=cuda ./task-latency-iris 1 10 1000   kernellaunch-multigpu-iris-cuda-${HOST}-10.csv
IRIS_ARCHS=cuda ./task-latency-iris 1 100 1000  kernellaunch-multigpu-iris-cuda-${HOST}-100.csv
IRIS_ARCHS=cuda ./task-latency-iris 1 1000 1000 kernellaunch-multigpu-iris-cuda-${HOST}-1000.csv

#and back to single-gpu iris
sed -i 's/#define IRIS_POOL_ENABLED     1/#define IRIS_POOL_ENABLED     0/g' ../../src/runtime/Pool.h
sed -i 's/#define IRIS_POOL_MAX_TASK    9999/#define IRIS_POOL_MAX_TASK    1100/g' ../../src/runtime/Pool.h
sed -i 's/#define IRIS_POOL_MAX_CMD     9999/#define IRIS_POOL_MAX_CMD     1100/g' ../../src/runtime/Pool.h

source ./setup.sh

