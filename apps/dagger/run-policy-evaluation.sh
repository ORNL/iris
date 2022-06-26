#!/bin/bash

#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
  echo "Aborting..."
  exit
fi

export SYSTEM=`hostname`

#start with a clean build of iris
cd ../.. ; ./build.sh ; cd apps/dagger
make clean
if [ "$SYSTEM" = "leconte" ] ; then
   module load gnu/9.2.0 nvhpc/21.3
   export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
   if [[ $PATH != *$CUDA_PATH* ]]; then
      export PATH=$CUDA_PATH/bin:$PATH
      export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
   fi
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "equinox" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip
else 
  rm -f *.csv ; make dagger_runner
fi

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f dagger_runner ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
  echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

echo "*******************************************************************"
echo "*                          Linear 10                              *"
echo "*******************************************************************"
#build linear-10 DAG
./dagger_generator.py --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json linear10-graph.json
#run -- all policy omitted because of deadlock
#for POLICY in random any
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 ; mv dagger_runner-$SYSTEM-*.csv linear-10-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done

# Parallel 2-by-10
echo "*******************************************************************"
echo "*                          Parallel 2by10                         *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --duplicates="2" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
cat graph.json
cp graph.json parallel-2by10-graph.json
[ $? -ne 0 ] && exit
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="2" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1; mv dagger_runner-$SYSTEM-*.csv parallel-2by10-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done

echo "*******************************************************************"
echo "*                          Parallel 5by100                        *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --duplicates="5" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json parallel-5by100-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk" --duplicates="5" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1; mv dagger_runner-$SYSTEM-*.csv parallel-5by100-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done

echo "*******************************************************************"
echo "*                          Diamond 10                             *"
echo "*******************************************************************"
#diamond 10
./dagger_generator.py --kernels="bigk" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --concurrent-kernels="bigk:3" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --sandwich
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json diamond-10-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --sandwich; mv dagger_runner-$SYSTEM-*.csv diamond-10-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done

echo "*******************************************************************"
echo "*                          Diamond 1000                           *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --concurrent-kernels="bigk:3" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --sandwich
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json diamond-1000-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --sandwich; mv dagger_runner-$SYSTEM-*.csv diamond-1000-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done

echo "*******************************************************************"
echo "*                          Chainlink 25                           *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --concurrent-kernels="bigk:3" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json chainlink-25-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --sandwich; mv dagger_runner-$SYSTEM-*.csv chainlink-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done
echo "*******************************************************************"
echo "*                          Galaga 25                             *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="bigk:3" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json galaga-25-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich; mv dagger_runner-$SYSTEM-*.csv chainlink-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done
echo "*******************************************************************"
echo "*                          Tangled 25                             *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="bigk:3" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json tangled-25-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich; mv dagger_runner-$SYSTEM-*.csv tangled-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done
echo "*******************************************************************"
echo "*                           Brain 1000                             *"
echo "*******************************************************************"
./dagger_generator.py --kernels="bigk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --concurrent-kernels="bigk:3" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10
[ $? -ne 0 ] && exit
cat graph.json
cp graph.json brain-1000-graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --sandwich; mv dagger_runner-$SYSTEM-*.csv brain-1000-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] && exit
done
#save
#rm -rf linear-10-results; mkdir -p linear-10-results; mv linear-10-*.csv linear-10-results
#echo "All results logged into ./linear-10-results"

