#!/bin/bash

source /auto/software/iris/setup_system.source

export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

if [ $MACHINE != "Zenith" ] ; then
  echo "Error: this test only works on the Zenith system. Exiting."
  exit
fi

export LD_LIBRARY_PATH=$IRIS_INSTALL_ROOT/lib:$LD_LIBRARY_PATH

make clean
make kernel.ptx kernel.hip test29_data_mem
[ $? -ne 0 ] && echo "Failed! Couldn't compile all kernels. Exiting." && exit 1

echo "Running OpenCL..."
IRIS_ARCHS=opencl ./test29_data_mem
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running CUDA..."
IRIS_ARCHS=cuda ./test29_data_mem
[ $? -ne 0 ] && echo "Failed! (CUDA backend) Exiting." && exit 1

echo "Running HIP..."
IRIS_ARCHS=hip ./test29_data_mem
[ $? -ne 0 ] && echo "Failed! (HIP backend) Exiting." && exit 1

module load gcc/12.1.0
make clean
make kernel.openmp.so test29_data_mem
[ $? -ne 0 ] && echo "Failed! Couldn't compile openmp kernels. Exiting." && exit 1
echo "Running (GNU) OpenMP..."
IRIS_ARCHS=openmp IRIS_KERNEL_BIN_OPENMP=`pwd`/kernel.openmp.so ./test29_data_mem
[ $? -ne 0 ] && echo "Failed! (OpenMP [GNU] backend) Exiting." && exit 1
module unload gcc/12.1.0

export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/cuda/11.7
make clean
make kernel.nvopenmp.so test29_data_mem
[ $? -ne 0 ] && echo "Failed! Couldn't compile openmp kernels. Exiting." && exit 1
echo "Running (NVIDIA) OpenMP..."
IRIS_ARCHS=openmp IRIS_KERNEL_BIN_OPENMP=`pwd`/kernel.nvopenmp.so ./test29_data_mem
[ $? -ne 0 ] && echo "Failed! (OpenMP [NVIDIA] backend) Exiting." && exit 1

exit 0
