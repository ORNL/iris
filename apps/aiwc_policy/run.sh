#!/bin/bash

# if zenith

source /auto/software/iris/setup_system.source
module load llvm/14.0.0
export IRIS_ARCHS=opencl
export LD_LIBRARY_PATH=$IRIS/lib:$LD_LIBRARY_PATH

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
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$AIWC -DBUILD_SHARED_LIBS=OFF ;
    [ $? -ne 0 ] && exit 1
    make -j$(nproc)
    [ $? -ne 0 ] && exit 1
    make install
    [ $? -ne 0 ] && exit 1
    echo "done."
fi
echo "AIWC found in: $AIWC"

make
[ $? -ne 0 ] && exit 1

echo "Running test with oclgrind device binary"
IRIS_ARCHS=opencl ~/.aiwc/bin/oclgrind --aiwc ./test_aiwc_policy
[ $? -ne 0 ] && exit 1
echo "Running test with oclgrind icd loader"
OCL_ICD_FILENAMES=$AIWC/lib/liboclgrind-rt-icd.so OCLGRIND_WORKLOAD_CHARACTERISATION=1 IRIS_ARCHS=opencl ./test_aiwc_policy
