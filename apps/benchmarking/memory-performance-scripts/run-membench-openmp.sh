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
make memory-performance-iris kernel.openmp.so

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f memory-performance-iris ] || ! [ -f kernel.openmp.so ] ; then
  exit
fi

export RUNTIME=openmp
export REPEATS=10
# Final experiment: Lock the number of transfers and increase the buffer-size---starting from 1KiB onwards
for SIZE in {1..25}
do
  ((ELEMENTS=2**${SIZE}))
  echo ${ELEMENTS}
  echo ${KIB}
  IRIS_ARCHS=openmp ./memory-performance-iris ${ELEMENTS} ${REPEATS} 1000 membench-${RUNTIME}-${HOST}-${ELEMENTS}.csv
done

source ./setup.sh

