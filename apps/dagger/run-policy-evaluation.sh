#!/bin/bash

#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
  echo "Aborting..."
  exit 1
fi

export SYSTEM=`hostname`
export RESULTS_DIR=`pwd`/dagger-figures
mkdir -p $RESULTS_DIR
echo "Running DAGGER evaluation.... (result figures can be found in $RESULTS_DIR)"

#start with a clean build of iris
cd ../.. ; ./build.sh && [ $? -ne 0 ] &&  exit 1 ; cd apps/dagger
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
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/22.7/cuda
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip
else 
  rm -f *.csv ; make dagger_runner
fi

# exit 1 if the last program run wasn't successful
[ $? -ne 0 ] &&  exit 1

#don't proceed if the target failed to build
if ! [ -f dagger_runner ] ; then
   exit 1
fi

#ensure libiris.so is in the shared library path
  echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

#Only run DAGGER once to generate the payloads to test the systems (we want to compare the scheduling algorithms over different systems, and so we should fix the payloads over the whole experiment)
#remove the dagger-payloads directory to regenerate payloads
if ! [ -d dagger-payloads ] ; then
  echo "Generating DAGGER payloads (delete this directory to regenerate new DAG payloads)..."
  mkdir -p dagger-payloads
  echo "*******************************************************************"
  echo "*                          Linear 10                              *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/linear10-graph.json
  cp dag.png $RESULTS_DIR/linear10-graph.png
  echo "*******************************************************************"
  echo "*                          Parallel 2by10                         *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/parallel-2by10-graph.json
  cp dag.png $RESULTS_DIR/parallel-2by10-graph.png
  echo "*******************************************************************"
  echo "*                          Parallel 5by100                        *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/parallel-5by100-graph.json
  cp dag.png $RESULTS_DIR/parallel-5by100-graph.png
  echo "*******************************************************************"
  echo "*                          Diamond 10                             *"
  echo "*******************************************************************"
  #diamond 10
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond-10-graph.json
  cp dag.png $RESULTS_DIR/diamond-10-graph.png
  echo "*******************************************************************"
  echo "*                          Diamond 100                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=100 --min-width=100 --max-width=100 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json diamond-100-graph.json
  cp dag.png $RESULTS_DIR/diamond-100-graph.png
  echo "*******************************************************************"
  echo "*                          Diamond 1000                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond-1000-graph.json
  cp dag.png $RESULTS_DIR/diamond-1000-graph.png
  echo "*******************************************************************"
  echo "*                          Chainlink 25                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/chainlink-25-graph.json
  cp dag.png $RESULTS_DIR/chainlink-25-graph.png
  echo "*******************************************************************"
  echo "*                          Galaga 25                              *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/galaga-25-graph.json
  cp dag.png $RESULTS_DIR/galaga-25-graph.png
  echo "*******************************************************************"
  echo "*                          Tangled 25                             *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/tangled-25-graph.json
  cp dag.png $RESULTS_DIR/tangled-25-graph.png
  echo "*******************************************************************"
  echo "*                           Brain 1000                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/brain-1000-graph.json
  cp dag.png $RESULTS_DIR/brain-1000-graph.png
fi

echo "Running DAGGER on payloads..."
echo "*******************************************************************"
echo "*                          Linear 10                              *"
echo "*******************************************************************"
##build linear-10 DAG
#./dagger_generator.py --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
#[ $? -ne 0 ] &&  exit 1
#cp graph.json linear10-graph.json
#cp dag.png $RESULTS_DIR/linear10-graph.png
#run -- all policy omitted because of deadlock
#for POLICY in random any
cp dagger-payloads/linear10-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] && echo "Linear 10 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/linear-10-$POLICY-$SYSTEM.csv
done

# Parallel 2-by-10
echo "*******************************************************************"
echo "*                          Parallel 2by10                         *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
#[ $? -ne 0 ] &&  exit 1
#cp graph.json parallel-2by10-graph.json
#cp dag.png $RESULTS_DIR/parallel-2by10-graph.png
cp dagger-payloads/parallel-2by10-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] && echo "Parallel 2by10 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/parallel-2by10-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Parallel 5by100                        *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
#[ $? -ne 0 ] &&  exit 1
#cp graph.json parallel-5by100-graph.json
#cp dag.png $RESULTS_DIR/parallel-5by100-graph.png
cp dagger-payloads/parallel-5by100-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
  [ $? -ne 0 ] && echo "Parallel 5by100 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/parallel-5by100-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Diamond 10                             *"
echo "*******************************************************************"
#diamond 10
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
#[ $? -ne 0 ] &&  exit 1
#cp graph.json diamond-10-graph.json
#cp dag.png $RESULTS_DIR/diamond-10-graph.png
cp dagger-payloads/diamond-10-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --sandwich
  [ $? -ne 0 ] && echo "Diamond 10 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/diamond-10-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Diamond 100                            *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=100 --min-width=100 --max-width=100 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
#[ $? -ne 0 ] &&  exit 1
#cp graph.json diamond-100-graph.json
#cp dag.png $RESULTS_DIR/diamond-100-graph.png
cp dagger-payloads/diamond-100-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=100 --min-width=100 --max-width=100 --sandwich
  [ $? -ne 0 ] && echo "Diamond 100 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/diamond-100-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Diamond 1000                           *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
#[ $? -ne 0 ] &&  exit 1
#cp graph.json diamond-1000-graph.json
#cp dag.png $RESULTS_DIR/diamond-1000-graph.png
cp dagger-payloads/diamond-1000-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --sandwich
  [ $? -ne 0 ] && echo "Diamond 1000 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/diamond-1000-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Chainlink 25                           *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
#[ $? -ne 0 ] &&  exit 1
#cp graph.json chainlink-25-graph.json
#cp dag.png $RESULTS_DIR/chainlink-25-graph.png
cp dagger-payloads/chainlink-25-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --sandwich
  [ $? -ne 0 ] && echo "Chainlink 25 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/chainlink-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Galaga 25                              *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
#[ $? -ne 0 ] &&  exit 1
#cp graph.json galaga-25-graph.json
#cp dag.png $RESULTS_DIR/galaga-25-graph.png
cp dagger-payloads/galaga-25-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich
  [ $? -ne 0 ] && echo "Galaga 25 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/chainlink-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                          Tangled 25                             *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3
#[ $? -ne 0 ] &&  exit 1
#cp graph.json tangled-25-graph.json
#cp dag.png $RESULTS_DIR/tangled-25-graph.png
cp dagger-payloads/tangled-25-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich
  [ $? -ne 0 ] && echo "Tangled 25 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/tangled-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

echo "*******************************************************************"
echo "*                           Brain 1000                            *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10
#[ $? -ne 0 ] &&  exit 1
#cp graph.json brain-1000-graph.json
#cp dag.png $RESULTS_DIR/brain-100-graph.png
cp dagger-payloads/brain-1000-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --sandwich
  [ $? -ne 0 ] && echo "Brain 1000 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM-*.csv $RESULTS_DIR/brain-1000-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
done

#TODO: add a mixed kernels test
#save
#rm -rf linear-10-results; mkdir -p linear-10-results; mv linear-10-*.csv linear-10-results
#echo "All results logged into ./linear-10-results"
exit 0
