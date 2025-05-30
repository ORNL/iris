#!/bin/bash

source /auto/software/iris/setup_system.source

export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

if [ $MACHINE != "Zenith" ] ; then
  echo "Error: this test only works on the Zenith system. Exiting."
  exit
fi

export LD_LIBRARY_PATH=$IRIS/lib:$LD_LIBRARY_PATH

make clean
make kernel.ptx kernel.hip test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! Couldn't compile all kernels. Exiting." && exit 1

echo "Running CUDA..."
IRIS_ARCHS=cuda ./test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! (CUDA backend) Exiting." && exit 1


echo "Running HIP..."
IRIS_ARCHS=hip ./test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! (HIP backend) Exiting." && exit 1


echo "Running (GNU) OpenMP..."
module load gcc/12.1.0
export OPENMP_PATH=/auto/software/swtree/ubuntu20.04/x86_64/gcc/12.1.0/lib64
make clean
make kernel.openmp.so test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! Couldn't compile openmp kernels. Exiting." && exit 1
IRIS_ARCHS=openmp IRIS_KERNEL_BIN_OPENMP=`pwd`/kernel.openmp.so ./test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! (OpenMP [GNU] backend) Exiting." && exit 1
module unload gcc/12.1.0


echo "Running (NVIDIA) OpenMP..."
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/cuda/11.7
make clean
make kernel.nvopenmp.so test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! Couldn't compile openmp kernels. Exiting." && exit 1
IRIS_ARCHS=openmp IRIS_KERNEL_BIN_OPENMP=`pwd`/kernel.nvopenmp.so ./test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! (OpenMP [NVIDIA] backend) Exiting." && exit 1


echo "Running OpenCL..."
IRIS_ARCHS=opencl ./test38_offset_subbuffer
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1


exit 0
