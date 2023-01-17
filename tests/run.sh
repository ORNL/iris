#!/bin/bash

#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
  echo "Aborting..."
  exit
fi
rm -rf build
export SYSTEM=`hostname`

#start with a clean build of iris
cd .. ; ./build.sh ; export iris_build_status="$?" ; cd iris-tests; mkdir build; cd build

if [ $iris_build_status -ne 0 ]; then
  exit
fi

if [ "$SYSTEM" = "leconte" ] ; then
   module load gnu/9.2.0 nvhpc/21.3
   export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
   if [[ $PATH != *$CUDA_PATH* ]]; then
      export PATH=$CUDA_PATH/bin:$PATH
      export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
   fi
elif [ "$SYSTEM" = "oswald*" ] ; then
  module load gnu/9.1.0 nvhpc/21.3 cmake/3.16.8
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
  if [[ $PATH != *$CUDA_PATH* ]]; then
     export PATH=$CUDA_PATH/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
  fi
elif [ "$SYSTEM" = "equinox" ] ; then
  export ROCM_PATH=/opt/rocm
  if [[ $PATH != *$ROCM_PATH* ]]; then
     export PATH=$ROCM_PATH/bin:$PATH
     export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
  fi
elif [ "$SYSTEM" = "explorer" ] ; then
  export ROCM_PATH=/opt/rocm
  if [[ $PATH != *$ROCM_PATH* ]]; then
     export PATH=$ROCM_PATH/bin:$PATH
     export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
  fi
fi

export IRIS_INSTALL_ROOT=$HOME/.local
#source ~/.local/setup.source
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib:$HOME/.local/lib64
cmake ..; make --ignore-errors; make test

