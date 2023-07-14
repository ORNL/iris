#!/bin/bash

#This script compares the performance of vector addition over increasing sized vectors (powers of 2**5..2**16) 
# initially cuda vs openmp in IRIS, then charm-sycl cuda vs iris cuda, finally dpc++ vs charm-sycl vs opensycl
REPEATS=25

execute_over_range () {
  for i in {5..16}
  do
    let SIZE="2**$i"
    $1 $SIZE $REPEATS $SIZE.csv;
    [ $? -ne 0 ] && exit
  done
}

make clean
# ensure we have all the binaries
make kernel.ptx kernel.openmp.so vecadd-iris vecadd-sycl vecadd-sycl-dpc++ vecadd-opensycl-openmp
[ $? -ne 0 ] && exit

#start with a clean log
rm -rf results; mkdir results;

# IRIS (OpenMP)
IRIS_ARCHS=openmp
execute_over_range "./vecadd-iris"
mkdir results/iris_openmp; mv *.csv results/iris_openmp

# IRIS (CUDA)
IRIS_ARCHS=cuda
execute_over_range "./vecadd-iris"
mkdir results/iris_cuda; mv *.csv results/iris_cuda

#charm-sycl (openmp)
export CHARM_SYCL_RTS=CPU
execute_over_range "./vecadd-sycl"
mkdir results/charmsycl_openmp_directly; mv *.csv results/charmsycl_openmp_directly

#charm-sycl (cuda)
export CHARM_SYCL_RTS=CUDA
execute_over_range "./vecadd-sycl"
mkdir results/charmsycl_cuda_directly; mv *.csv results/charmsycl_cuda_directly

#charm-sycl (openmp)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=openmp
execute_over_range "./vecadd-sycl"
mkdir results/charmsycl_openmp; mv *.csv results/charmsycl_openmp

#charm-sycl (cuda)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=cuda
execute_over_range "./vecadd-sycl"
mkdir results/charmsycl_cuda; mv *.csv results/charmsycl_cuda

#dpc++ (cuda)
export DPCPP_HOME=$HOME/dpc++-workspace
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
execute_over_range "./vecadd-sycl-dpc++"
mkdir results/dpc++_cuda; mv *.csv results/dpc++_cuda

#opensycl (openmp)
execute_over_range "./vecadd-opensycl-openmp"
mkdir results/opensycl_openmp; mv *.csv results/opensycl_openmp

##opensycl (cuda)
#execute_over_range "./vecadd-opensycl-cuda"
#mkdir results/opensycl_cuda; mv *.csv results/opensycl_cuda

