#!/bin/bash

#This script compares the performance of empty tasks but increasing sized memory objects (powers of 2**5..2**16) 
# initially cuda vs openmp in IRIS, then charm-sycl cuda vs iris cuda, finally dpc++ vs charm-sycl vs opensycl
REPEATS=100

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
make kernel.ptx kernel.openmp.so kernel.hip empty-iris empty-sycl empty-sycl-dpc++ empty-opensycl-openmp
[ $? -ne 0 ] && exit

#start with a clean log
rm -rf results/empty; mkdir -p results ; mkdir results/empty

# IRIS (OpenMP)
export IRIS_ARCHS=openmp
execute_over_range "./empty-iris"
mkdir results/empty/iris_openmp; mv *.csv results/empty/iris_openmp

# IRIS (CUDA)
export IRIS_ARCHS=cuda
execute_over_range "./empty-iris"
mkdir results/empty/iris_cuda; mv *.csv results/empty/iris_cuda

# IRIS (HIP)
export IRIS_ARCHS=hip
execute_over_range "./empty-iris"
mkdir results/empty/iris_hip; mv *.csv results/empty/iris_hip

#charm-sycl (openmp)
export CHARM_SYCL_RTS=CPU
execute_over_range "./empty-sycl"
mkdir results/empty/charmsycl_openmp_directly; mv *.csv results/empty/charmsycl_openmp_directly

#charm-sycl (cuda)
export CHARM_SYCL_RTS=CUDA
execute_over_range "./empty-sycl"
mkdir results/empty/charmsycl_cuda_directly; mv *.csv results/empty/charmsycl_cuda_directly

#charm-sycl (hip)
export CHARM_SYCL_RTS=HIP
execute_over_range "./empty-sycl"
mkdir results/empty/charmsycl_hip_directly; mv *.csv results/empty/charmsycl_hip_directly

#charm-sycl (openmp)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=openmp
execute_over_range "./empty-sycl"
mkdir results/empty/charmsycl_openmp; mv *.csv results/empty/charmsycl_openmp

#charm-sycl (cuda)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=cuda
execute_over_range "./empty-sycl"
mkdir results/empty/charmsycl_cuda; mv *.csv results/empty/charmsycl_cuda

#charm-sycl (hip)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=hip
execute_over_range "./empty-sycl"
mkdir results/empty/charmsycl_hip; mv *.csv results/empty/charmsycl_hip

#dpc++ (cuda)
export DPCPP_HOME=$HOME/dpc++-workspace
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
execute_over_range "./empty-sycl-dpc++"
mkdir results/empty/dpc++_cuda; mv *.csv results/empty/dpc++_cuda

#opensycl (openmp)
execute_over_range "./empty-opensycl-openmp"
mkdir results/empty/opensycl_openmp; mv *.csv results/empty/opensycl_openmp

##opensycl (cuda)
#execute_over_range "./empty-opensycl-cuda"
#mkdir results/empty/opensycl_cuda; mv *.csv results/empty/opensycl_cuda

