#!/bin/bash

#This script compares the performance of vector addition over increasing sized vectors (powers of 2**5..2**16) 
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

# Optimization tests -- can we get this for free by adding IRIS-data memory to CharmSYCL?
mkdir -p results;
make kernel.ptx kernel.hip kernel.openmp.so vecadd-iris-data-memory vecadd-sycl-discard-write vecadd-sycl-dpc++-discard-write vecadd-opensycl-openmp-discard-write
[ $? -ne 0 ] && exit

# IRIS (OpenMP) using data-memory
IRIS_ARCHS=openmp
execute_over_range "./vecadd-iris-data-memory"
mkdir results/iris_openmp_data_memory; mv *.csv results/iris_openmp_data_memory

# IRIS (CUDA)
IRIS_ARCHS=cuda
execute_over_range "./vecadd-iris-data-memory"
mkdir results/iris_cuda_data_memory; mv *.csv results/iris_cuda_data_memory

# IRIS (HIP)
IRIS_ARCHS=hip
execute_over_range "./vecadd-iris-data-memory"
mkdir results/iris_hip_data_memory; mv *.csv results/iris_hip_data_memory

#charm-sycl (openmp)
export CHARM_SYCL_RTS=CPU
execute_over_range "./vecadd-sycl-discard-write"
mkdir results/charmsycl_openmp_directly_discard_write; mv *.csv results/charmsycl_openmp_directly_discard_write

#charm-sycl (cuda)
export CHARM_SYCL_RTS=CUDA
execute_over_range "./vecadd-sycl-discard-write"
mkdir results/charmsycl_cuda_directly_discard_write; mv *.csv results/charmsycl_cuda_directly_discard_write

#charm-sycl (hip)
export CHARM_SYCL_RTS=HIP
execute_over_range "./vecadd-sycl-discard-write"
mkdir results/charmsycl_hip_directly_discard_write; mv *.csv results/charmsycl_hip_directly_discard_write

#charm-sycl (IRIS openmp)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=openmp
execute_over_range "./vecadd-sycl-discard-write"
mkdir results/charmsycl_openmp_discard_write; mv *.csv results/charmsycl_openmp_discard_write

#charm-sycl (IRIS cuda)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=cuda
execute_over_range "./vecadd-sycl-discard-write"
mkdir results/charmsycl_cuda_discard_write; mv *.csv results/charmsycl_cuda_discard_write

#charm-sycl (IRIS hip)
export CHARM_SYCL_RTS=IRIS
#export CHARM_SYCL_IRIS_POLICY=all -- zenith only has one device
export IRIS_ARCHS=hip
execute_over_range "./vecadd-sycl-discard-write"
mkdir results/charmsycl_hip_discard_write; mv *.csv results/charmsycl_hip_discard_write

#dpc++ (cuda)
export DPCPP_HOME=$HOME/dpc++-workspace
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
execute_over_range "./vecadd-sycl-dpc++-discard-write"
mkdir results/dpc++_cuda_discard_write; mv *.csv results/dpc++_cuda_discard_write

#opensycl (openmp)
execute_over_range "./vecadd-opensycl-openmp-discard-write"
mkdir results/opensycl_openmp_discard_write; mv *.csv results/opensycl_openmp_discard_write

##opensycl (cuda)
#execute_over_range "./vecadd-opensycl-cuda-discard-write"
#mkdir results/opensycl_cuda_discard_write; mv *.csv results/opensycl_cuda_discard_write

