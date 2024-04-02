#!/bin/bash
export SKIP_SETUP=${SKIP_SETUP:=0}
set -x;
if [ "x$SKIP_SETUP" = "x0" ]; then
source ./setup.sh
fi

if [ "$SYSTEM" = "leconte" ] ; then
  rm -f *.csv ; make dagger_test kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "equinox" ] ; then
  rm -f *.csv ; make dagger_test kernel.ptx kernel.openmp.so
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_test kernel.hip kernel.openmp.so
elif [ "$SYSTEM" = "radeon" ] ; then
  rm -f *.csv ; make dagger_test kernel.hip kernel.openmp.so
elif [ "$SYSTEM" = "zenith" ] ; then
  rm -f *.csv ; make dagger_test kernel.hip kernel.ptx kernel.openmp.so
else
  echo "Unknown system." && exit 1
fi

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit $?

#don't proceed if the target failed to build
if ! [ -f dagger_test ] ; then
  exit 1
fi

#ensure libiris.so is in the shared library path
#echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH
echo "*******************************************************************"
echo "*                          Linear 50                              *"
echo "*******************************************************************"
##build linear-50 DAG
./dagger_generator.py --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1 --concurrent-kernels="ijk:1"
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json linear50-graph.json
for POLICY in roundrobin depend profile random ftf sdq
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_test --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1

  [ $? -ne 0 ] && exit
done
exit 0

#echo "*******************************************************************"
#echo "*                          Linear 50x3                            *"
#echo "*******************************************************************"
##build linear-50 DAG
#./dagger_generator.py --kernels="ijk" --duplicates="3" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#[ $? -ne 0 ] && exit
#cat graph.json
#cp graph.json linear50x3-graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_test --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="3" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#  [ $? -ne 0 ] && exit
#done
#
#echo "*******************************************************************"
#echo "*                          Linear 50x8                            *"
#echo "*******************************************************************"
##build linear-50 DAG
#./dagger_generator.py --kernels="ijk" --duplicates="8" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#[ $? -ne 0 ] && exit
#cat graph.json
#cp graph.json linear50x8-graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_test --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="8" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
##  IRIS_HISTORY=1 gdb --args ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="3" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#  [ $? -ne 0 ] && exit
#done
