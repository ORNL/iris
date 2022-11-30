#!/bin/bash
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

export IRIS_SRC_DIR=../..
export WORKING_DIR=`pwd`
export SYSTEM=`hostname`

#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
  echo "Aborting..."
  exit
fi

#don't use conda's version of gcc (and libc)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

#start with a clean build of iris
rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
cd $IRIS_SRC_DIR ; ./build.sh; [ $? -ne 0 ] && exit ; cd $WORKING_DIR ;
make clean
if [ "$SYSTEM" = "leconte" ] ; then
   module load gnu/9.2.0 nvhpc/21.3
   export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
   if [[ $PATH != *$CUDA_PATH* ]]; then
      export PATH=$CUDA_PATH/bin:$PATH
      export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
   fi
elif [ "$SYSTEM" = "equinox.ftpn.ornl.gov" ] ; then
  #export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/22.7/cuda
  export SYSTEM="equinox"
elif [ "$SYSTEM" = "zenith.ftpn.ornl.gov" ] ; then
  export SYSTEM="zenith"
fi

source $IRIS_INSTALL_ROOT/setup.source

