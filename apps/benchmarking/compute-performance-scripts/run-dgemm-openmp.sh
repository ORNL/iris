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
make compute-performance-iris kernel.openmp.so

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f compute-performance-iris ] || ! [ -f kernel.openmp.so ] ; then
  exit
fi

#<application name> <memory size> <verbose?> <number of devices to use> <number of statistical samples> <log file to store samples>
#run openmp baseline to see FLOP scaling over increasing device count
REPEATS=1
#for num_devices in {6..6}
#REPEATS=100
for num_devices in {1..13}
do
    IRIS_ARCHS=openmp ./compute-performance-iris 4096 0 ${num_devices} ${REPEATS} dgemm-iris-openmp-${HOST}-${num_devices}.csv
done

source ./setup.sh

