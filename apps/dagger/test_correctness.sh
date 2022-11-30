#!/bin/bash
source ./setup.sh

if [ "$SYSTEM" = "leconte" ] ; then
  rm -f *.csv ; make dagger_test kernel.ptx
elif [ "$SYSTEM" = "equinox" ] ; then
  rm -f *.csv ; make dagger_test kernel.ptx
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_test kernel.hip
elif [ "$SYSTEM" = "radeon" ] ; then
  rm -f *.csv ; make dagger_test kernel.hip
elif [ "$SYSTEM" = "zenith" ] ; then
  rm -f *.csv ; make dagger_test kernel.hip kernel.ptx
else
  echo "Unknown system." && exit
fi

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f dagger_test ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
  echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

echo "*******************************************************************"
echo "*                          Linear 50                              *"
echo "*******************************************************************"
##build linear-50 DAG
./dagger_generator.py --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json linear50-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_test --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1

  [ $? -ne 0 ] && exit
done

#echo "*******************************************************************"
#echo "*                          Linear 50x3                            *"
#echo "*******************************************************************"
##build linear-50 DAG
#./dagger_generator.py --kernels="bigk" --duplicates="3" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#[ $? -ne 0 ] && exit
#cat graph.json
#cp graph.json linear50x3-graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_test --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="3" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#  [ $? -ne 0 ] && exit
#done
#
#echo "*******************************************************************"
#echo "*                          Linear 50x8                            *"
#echo "*******************************************************************"
##build linear-50 DAG
#./dagger_generator.py --kernels="bigk" --duplicates="8" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#[ $? -ne 0 ] && exit
#cat graph.json
#cp graph.json linear50x8-graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_test --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="8" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
##  IRIS_HISTORY=1 gdb --args ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="3" --buffers-per-kernel="bigk:rw r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=50 --num-tasks=50 --min-width=1 --max-width=1
#  [ $? -ne 0 ] && exit
#done
