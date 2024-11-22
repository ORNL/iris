#!/bin/bash
set -x;
source /auto/software/iris/setup_system.source

export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

if [ $MACHINE != "Zenith" ] ; then
  echo "Error: this test only works on the Zenith system. Exiting."
  exit
fi

export LD_LIBRARY_PATH=$IRIS_INSTALL_ROOT/lib:$LD_LIBRARY_PATH

make clean
make test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! Couldn't compile all kernels. Exiting." && exit 1

echo "Running OpenCL... (default)"
IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OPENCL_VENDOR_PATH=$CUDA_PATH/lib64/libOpenCL.so)"
OPENCL_VENDOR_PATH=$CUDA_PATH/lib64/libOpenCL.so IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OPENCL_VENDOR_PATH=$ROCM_PATH/lib/libOpenCL.so)"
OPENCL_VENDOR_PATH=$ROCM_PATH/lib/libOpenCL.so IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OPENCL_VENDOR_PATH=$OPENCL_PATH/lib/libamdocl64.so)"
OPENCL_VENDOR_PATH=$OPENCL_PATH/lib/libamdocl64.so IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OCL_ICD_VENDORS=$CUDA_PATH/lib64/)"
OCL_ICD_VENDORS=$CUDA_PATH/lib64/ IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OCL_ICD_VENDORS=$CUDA_PATH/lib64)"
OCL_ICD_VENDORS=$CUDA_PATH/lib64 IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OCL_ICD_VENDORS=$CUDA_PATH/lib64/ OCL_ICD_FILENAMES=libOpenCL.so)"
OCL_ICD_VENDORS=$CUDA_PATH/lib64/ OCL_ICD_FILENAMES=libOpenCL.so IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OCL_ICD_FILENAMES=$CUDA_PATH/lib64/libOpenCL.so)"
OCL_ICD_FILENAMES=$CUDA_PATH/lib64/libOpenCL.so IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

echo "Running OpenCL... (OCL_ICD_VENDORS=$CUDA_PATH/lib64 OCL_ICD_FILENAMES=libOpenCL.so)"
OCL_ICD_VENDORS=$CUDA_PATH/lib64 OCL_ICD_FILENAMES=libOpenCL.so IRIS_ARCHS=opencl ./test37_opencl_icd
[ $? -ne 0 ] && echo "Failed! (OpenCL backend) Exiting." && exit 1

#Oclgrind
if [[ -z "${AIWC}" ]] ; then
  export AIWC_INSTALL_ROOT=$HOME/.aiwc
  export AIWC=$AIWC_INSTALL_ROOT
fi
module load llvm/13.0.1
export LLVM_ROOT=/auto/software/swtree/ubuntu20.04/x86_64/llvm/13.0.1
#confirm we have Oclgrind with AIWC installed
echo "Looking for AIWC in: $AIWC ..."
if [ ! -d "$AIWC" ] 
then
    echo "Not found. Installing to: $AIWC..."
    export WORKING_DIR=`pwd`
    export OCLGRIND_SRC=aiwc-src
    git clone https://github.com/BeauJoh/AIWC.git $OCLGRIND_SRC
    [ $? -ne 0 ] && exit 1
    mkdir $OCLGRIND_SRC/build ;
    [ $? -ne 0 ] && exit 1
    cd $OCLGRIND_SRC/build ;
    [ $? -ne 0 ] && exit 1
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$AIWC -DBUILD_SHARED_LIBS=ON -DLLVM_ROOT_DIR=$LLVM_ROOT;
    [ $? -ne 0 ] && exit 1
    make -j$(nproc)
    [ $? -ne 0 ] && exit 1
    make install
    [ $? -ne 0 ] && exit 1
    echo "done."
    cd $WORKING_DIR
fi
echo "AIWC found in: $AIWC"

echo "Running OpenCL... (OclGrind)"
OCL_ICD_VENDORS=$AIWC/lib/ OCL_ICD_FILENAMES=liboclgrind-rt.so IRIS_ARCHS=opencl ./test37_opencl_icd
exit 0
