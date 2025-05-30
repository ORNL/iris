#!/bin/bash

#uncomment to use data memory rather than explicit memory transfers
export USE_DATA_MEMORY=${USE_DATA_MEMORY:="--use-data-memory"}
export REPEATS=${REPEATS:=100}
export PAYLOAD_SIZE=${PAYLOAD_SIZE:=128}
export SKIP_SETUP=${SKIP_SETUP:=0}

set -x;
if [ "x$SKIP_SETUP" = "x0" ]; then
source ./setup.sh
[ $? -ne 0 ] &&  exit 1
fi

make clean
if [ "$SYSTEM" = "leconte" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "equinox" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "oswald" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.openmp.so
elif [ "$SYSTEM" = "radeon" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.openmp.so
elif [ "$SYSTEM" = "zenith" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "orc-open-hyp" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "milan" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "milan2" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "hudson" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx kernel.openmp.so
else
   echo "Unknown system ($SYSTEM)." && exit 1
fi

# exit 1 if the last program run wasn't successful
[ $? -ne 0 ] &&  exit 1

#don't proceed if the target failed to build
if ! [ -f dagger_runner ] ; then
   echo "No dagger_runner app! " && exit 1
fi


