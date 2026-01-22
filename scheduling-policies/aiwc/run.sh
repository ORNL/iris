#!/bin/bash

source /auto/software/iris/setup_system.source
module load llvm/14.0.0
export IRIS_ARCHS=opencl
export LD_LIBRARY_PATH=$IRIS/lib:$LD_LIBRARY_PATH
export WORKING_DIR=`pwd`

if [[ -z "${AIWC}" ]] ; then
  export AIWC_INSTALL_ROOT=$HOME/.aiwc
  export AIWC=$AIWC_INSTALL_ROOT
fi
#confirm we have Oclgrind with AIWC installed
echo "Looking for AIWC in: $AIWC ..."
if [ ! -d "$AIWC" ] 
then
    echo "Not found. Installing to: $AIWC..."
    export OCLGRIND_SRC=aiwc-src
    git clone https://github.com/BeauJoh/AIWC.git $OCLGRIND_SRC
    [ $? -ne 0 ] && exit 1
    mkdir $OCLGRIND_SRC/build ;
    [ $? -ne 0 ] && exit 1
    cd $OCLGRIND_SRC/build ;
    [ $? -ne 0 ] && exit 1
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$AIWC -DBUILD_SHARED_LIBS=ON ;
    [ $? -ne 0 ] && exit 1
    make -j$(nproc)
    [ $? -ne 0 ] && exit 1
    make install
    [ $? -ne 0 ] && exit 1
    echo "done."
    cd $WORKING_DIR
fi
echo "AIWC found in: $AIWC"

make
[ $? -ne 0 ] && exit 1

echo "Running test with oclgrind device binary"
IRIS_ARCHS=opencl $AIWC/bin/oclgrind --aiwc ./test_aiwc_policy
[ $? -ne 0 ] && exit 1
echo "Running test with oclgrind icd loader"
OPENCL_VENDOR_PATH=$AIWC/lib OCL_ICD_VENDORS=liboclgrind-rt.so OCLGRIND_WORKLOAD_CHARACTERISATION=1 IRIS_ARCHS=opencl ./test_aiwc_policy
[ $? -ne 0 ] && exit 1

