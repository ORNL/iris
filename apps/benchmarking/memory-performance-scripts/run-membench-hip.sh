#!/bin/bash
source ./setup.sh

export WORKING_DIR=`pwd`

make clean
make memory-performance-iris-profiling kernel.hip

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f memory-performance-iris-profiling ] || ! [ -f kernel.hip ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

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

export RUNTIME=hip
# Final experiment: Lock the number of transfers and increase the buffer-size---starting from 1KiB onwards
for SIZE in {1..25}
do
  ((ELEMENTS=2**${SIZE}))
  echo ${ELEMENTS}
  echo ${KIB}
  #gdb --args ./membench-iris-profiling ${ELEMENTS} 1000 1000 membench-${RUNTIME}-${MACHINE}-${ELEMENTS}.csv
  IRIS_ARCHS=hip ./memory-performance-iris-profiling ${ELEMENTS} 1000 1000 membench-${RUNTIME}-${MACHINE}-${ELEMENTS}.csv
done


#mkdir -p extensive-transfer-size-results && mv membench-*.csv extensive-transfer-size-results


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
