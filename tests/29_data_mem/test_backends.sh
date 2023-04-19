#!/bin/bash

source /auto/software/iris/setup_system.source
export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

if [ $MACHINE != "Zenith" ] ; then
  echo "Error: this test only works on the Zenith system. Exiting."
  exit
fi

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

make clean
make kernel.openmp.so kernel.ptx kernel.hip test29_data_mem
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

echo "Running OpenMP..."
IRIS_ARCHS=openmp ./test29_data_mem
[ $? -ne 0 ] && echo "Failed! (OpenMP backend) Exiting." && exit 1
