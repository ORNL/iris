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
make memory-performance-iris kernel.ptx

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f memory-performance-iris ] || ! [ -f kernel.ptx ] ; then
  exit
fi

export RUNTIME=cuda
export REPEATS=25

# Final experiment: Lock the number of transfers and increase the buffer-size---starting from 1KiB onwards
for SIZE in {1..26}
do
  ((ELEMENTS=2**${SIZE}))
  echo ${ELEMENTS}
  echo ${KIB}
  IRIS_ARCHS=cuda ./memory-performance-iris ${ELEMENTS} ${REPEATS} 1000 membench-${RUNTIME}-${HOST}-${ELEMENTS}.csv
done

source ./setup.sh

# Lock the number of transfers and increase the buffer-size---starting from 1KiB onwards
#for SIZE in {1..25}
#do
#  ((ELEMENTS=2**${SIZE}))
#  echo ${ELEMENTS}
#  echo ${KIB}
#  ./membench-iris-profiling ${ELEMENTS} 1 1000 membench-${RUNTIME}-${MACHINE}-${ELEMENTS}.csv
#done


#and run an outlier point to compare the accuracy of the prediction
#for SIZE in {30..30}
#do
#  ((ELEMENTS=2**${SIZE}))
#  echo ${ELEMENTS}
#  echo ${KIB}
#  ./membench-iris-profiling ${ELEMENTS} 1 1000 membench-${RUNTIME}-${MACHINE}-${ELEMENTS}.csv
#done

## Lock the number of transfers and increase the buffer-size---starting from 1KiB onwards
#for SIZE in {8..18}
#do
#  ((ELEMENTS=2**${SIZE}))
#  ((KIB=(${ELEMENTS}*4)/1024))
#  echo ${ELEMENTS}
#  echo ${KIB}
#  ./membench-iris-profiling ${ELEMENTS} 1 100 membench-cuda-${MACHINE}-${KIB}KiB-10.csv
#done
#
#mkdir -p transfer-size-results && mv membench-*.csv transfer-size-results


## Lock the buffer size but adjust the number of transfers (the length of the task-chain)
#for LEN in 1 10 100 1000
#do
#  ./membench-iris-profiling 1024 ${LEN} 100 membench-cuda-${MACHINE}-1024-${LEN}.csv
#done
#
#mkdir -p number-of-transfers-results && mv membench-*.csv number-of-transfers-results


