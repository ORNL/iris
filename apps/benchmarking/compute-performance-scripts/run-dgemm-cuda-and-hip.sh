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
make compute-performance-iris kernel.ptx kernel.hip

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f compute-performance-iris ] || ! [ -f kernel.ptx ] || ! [ -f kernel.hip ] ; then
  exit
fi

#<application name> <memory size> <verbose?> <number of devices to use> <number of statistical samples> <log file to store samples>
#run CUDA baseline to see FLOP scaling over increasing device count
export HOST=zenith
SIZE=2048
REPEATS=10
VERIFY=0


#for num_devices in {1..13}
#do
#    IRIS_ARCHS=cuda,hip IRIS_DEV_ENABLED=1 ./compute-performance-iris ${SIZE} ${VERIFY} ${num_devices} ${REPEATS} dgemm-iris-cuda+hip-${HOST}-${num_devices}.csv
#done
for num_devices in {1..13}
do
  for (( num_run=0; num_run<=$REPEATS; num_run++ ))
  do
    IRIS_ARCHS=cuda,hip ./compute-performance-iris ${SIZE} ${VERIFY} ${num_devices} ${REPEATS} dgemm-iris-cuda+hip-${HOST}-${num_devices}.csv
  done
done



source ./setup.sh

