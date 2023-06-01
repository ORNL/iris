#!/bin/bash
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

export IRIS_SRC_DIR=../..
export WORKING_DIR=`pwd`
export SYSTEM=`hostname`

#if we don't have a conda env set, then load it.
#if [[ -z "$CONDA_PREFIX" ]] ; then
#  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
#  echo "Aborting..."
#  exit
#fi

#don't use conda's version of gcc (and libc)
#export CC=/usr/bin/gcc
#export CXX=/usr/bin/g++

if [ "$SYSTEM" = "leconte" ] ; then
  export SYSTEM="leconte"
  export MACHINE=Leconte
  module load cmake gnu/9.2.0 nvhpc/22.11
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/22.11/cuda/11.0
  export PATH=$CUDA_PATH/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH
  export NVCC_FLAGS="-arch=sm_70" #V100
elif [ "$SYSTEM" = "oswald00" ] ; then
  export SYSTEM="oswald"
  export MACHINE=Oswald
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/cuda
  export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH
  export NVCC_FLAGS="-arch=sm_60" #P100
  export OPENCL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/cuda
elif [ "$SYSTEM" = "equinox.ftpn.ornl.gov" ] ; then
  export SYSTEM="equinox"
  export MACHINE=Equinox
  module load cmake gnu/9.4.0 nvhpc/22.11
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7
  export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH
  export NVCC_FLAGS="-arch=sm_70" #V100
  export OPENCL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7
elif [ "$SYSTEM" = "zenith.ftpn.ornl.gov" ] ; then
  export SYSTEM="zenith"
  export MACHINE=Zenith
  module load cmake nvhpc/22.9
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/cuda/11.7
  export NVCC_FLAGS="-arch=sm_86" #GA102
  export ROCM_PATH=/opt/rocm
  export OPENCL_PATH=/opt/rocm/opencl
elif [ "$SYSTEM" = "explorer" ] ; then
  export SYSTEM="explorer"
  export MACHINE=Explorer
  module load cmake
  export ROCM_PATH=/opt/rocm
  export OPENCL_PATH=/opt/rocm/opencl
elif [ "$SYSTEM" = "radeon.ftpn.ornl.gov" ] ; then
  export SYSTEM="radeon"
  export MACHINE=Radeon
  module load cmake
  export ROCM_PATH=/opt/rocm
  export OPENCL_PATH=/opt/rocm/opencl
fi

source $IRIS_INSTALL_ROOT/setup.source

#start with a clean build of iris
rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
cd $IRIS_SRC_DIR ; ./build.sh; [ $? -ne 0 ] && exit ; cd $WORKING_DIR ;

