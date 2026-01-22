#!/bin/bash
source /auto/software/iris/setup_system.source

export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

echo MACHINE = $MACHINE SYSTEM is $SYSTEM
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

export IRIS_SRC_DIR=../..
export WORKING_DIR=`pwd`

#installed with:
#micromamba create -f dagger.yaml
#micromamba activate dagger
#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: mamba activate dagger)"
  echo "Aborting..."
  exit
fi

#extra sourcing
if [ "$SYSTEM" = "leconte" ] ; then
  export LD_LIBRARY_PATH=$HOME/.iris/lib64:$LD_LIBRARY_PATH
elif [ "$SYSTEM" = "equinox" ] ; then
  module load gnu/12.2.0
elif [ "$SYSTEM" = "oswald" ] ; then
  module load cmake nvhpc/23.7 gcc/12.1.0 #gnu/8.3.0 #gnu/3-2a
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.iris/lib64
  export NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/bin/nvcc
  export CC=gcc
  export CXX=g++
fi

#start with a clean build of iris
echo "IRIS source directory is: $IRIS_SRC_DIR"
echo "working directory is: $WORKING_DIR"
rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
cd $IRIS_SRC_DIR ; ./build.sh && [ $? -ne 0 ] && exit ; cd $WORKING_DIR ;

source $IRIS_INSTALL_ROOT/setup.source
make clean

export IRIS_HISTORY=1
#export IRIS_ARCHS=opencl
export IRIS_ARCHS=cuda,hip
echo TODO: try out opencl to verify the explicit memory transfer huge performance penalty persists on all but CUDA devices with explicit memory management but disappears with the use of data-memory
