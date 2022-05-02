#!/bin/bash

#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
  echo "Aborting..."
  exit
fi

# load system specific modules
export SYSTEM=`hostname`
if [[ "$SYSTEM" = oswald* ]] ; then
  module load cmake/3.19.2 gnu/9.1.0 nvhpc/21.3
  export SM=sm_60
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda
  #only update the path and library-path the first time!
  if [[ $PATH != *$CUDA_PATH* ]]; then
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
  fi

  export TARGETS="dagger_runner kernel.ptx"
elif [[ "$SYSTEM" = leconte ]] ; then
   module load gnu/9.2.0 nvhpc/21.3
   export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
   if [[ $PATH != *$CUDA_PATH* ]]; then
      export PATH=$CUDA_PATH/bin:$PATH
      export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
   fi

   export TARGETS="dagger_runner kernel.ptx"
elif [[ "$SYSTEM" = explorer ]] ; then
  export TARGETS="dagger_runner kernel.hip"
else
  export TARGETS="dagger_runner"
fi

#start with a clean build of iris
cd ../.. ; ./build.sh ; export iris_build_status="$?" ; cd apps/dagger
#and abort if iris didn't successfully install
if [ $iris_build_status -ne 0 ]; then
  exit
fi

#build app
make clean
rm -f *.csv ; echo "making targets: $TARGETS"; make $TARGETS
#don't proceed if the target failed to build
if [ "$?" -ne 0 ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
if [[ $PATH != *$HOME/.local/lib* ]]; then
  echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH
fi

#build linear-10 DAG
./dagger_generator.py --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
#run -- all policy omitted because of deadlock
for POLICY in roundrobin depend profile random any
#for POLICY in roundrobin depend profile random any all
do
  IRIS_HISTORY=1 ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=1024  --kernels="bigk" --duplicates="0" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 ; mv dagger_runner-$SYSTEM-*.csv linear-10-$POLICY.csv
done
#save
rm -rf linear-10-results; mkdir -p linear-10-results; mv linear-10-*.csv linear-10-results
echo "All results logged into ./linear-10-results"

##build diamond-10 DAG
#./dagger_generator.py --kernels="bigk" --duplicates="10" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=3 --num-tasks=1 --min-width=1 --max-width=1 --sandwich
##run -- all policy omitted because of deadlock
#for POLICY in roundrobin depend profile random any
##for POLICY in roundrobin depend profile random any all
#do
#  ./dagger_runner --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=1024  --kernels="bigk" --duplicates="10" --buffers-per-kernel="bigk:w r r" --kernel-dimensions="bigk:2" --kernel-split='100' --depth=3 --num-tasks=1 --min-width=1 --max-width=1 --sandwich; mv dagger_runner-$SYSTEM-*.csv diamond-10-$POLICY.csv
#done
##save
#rm -rf diamond-10-results; mkdir -p diamond-10-results; mv diamond-10-*.csv diamond-10-results
#echo "All results logged into ./diamond-10-results"


