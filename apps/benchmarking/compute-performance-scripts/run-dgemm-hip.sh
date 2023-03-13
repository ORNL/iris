#!/bin/bash

source ./setup.sh

export WORKING_DIR=`pwd`

make clean
make compute-performance-iris-profiling kernel.hip

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f compute-performance-iris-profiling ] || ! [ -f kernel.hip ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

##<application name> <memory size> <verbose?> <number of devices to use> <number of statistical samples> <log file to store samples>
##run HIP baseline to see FLOP scaling over increasing device count
REPEATS=1
#for num_devices in {6..6}
#REPEATS=100
for num_devices in {1..13}
do
    IRIS_ARCHS=hip ./compute-performance-iris-profiling 4096 0 ${num_devices} ${REPEATS} dgemm-iris-hip-${MACHINE}-${num_devices}.csv
done

#run IRIS multi-device
#multiple GPU requires pooling to be enabled in iris:
sed -i 's/#define BRISBANE_POOL_ENABLED     0/#define BRISBANE_POOL_ENABLED     1/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_TASK    1100/#define BRISBANE_POOL_MAX_TASK    9999/g' ../../src/runtime/Pool.h
sed -i 's/#define BRISBANE_POOL_MAX_CMD     1100/#define BRISBANE_POOL_MAX_CMD     9999/g' ../../src/runtime/Pool.h

source ./setup.sh

#for num_devices in {1..13}
#do
#    IRIS_ARCHS=hip ./compute-performance-iris-profiling 4096 0 ${num_devices} ${REPEATS} dgemm-iris-hip-${MACHINE}-${num_devices}.csv
#done
#
##and back to single-gpu iris
#sed -i 's/#define BRISBANE_POOL_ENABLED     1/#define BRISBANE_POOL_ENABLED     0/g' ../../src/runtime/Pool.h
#sed -i 's/#define BRISBANE_POOL_MAX_TASK    9999/#define BRISBANE_POOL_MAX_TASK    1100/g' ../../src/runtime/Pool.h
#sed -i 's/#define BRISBANE_POOL_MAX_CMD     9999/#define BRISBANE_POOL_MAX_CMD     1100/g' ../../src/runtime/Pool.h
#
#source ./setup.sh
#
