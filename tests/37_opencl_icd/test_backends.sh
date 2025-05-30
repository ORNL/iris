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
latest_version=$(module avail llvm 2>&1 | grep -o 'llvm/[0-9]\+\.[0-9]\+\.[0-9][_a-zA-Z0-9\-]*' | grep -v "_.*" | awk -F'/' '{print $2}' | sort -V | grep -E '^(19\.0\.1|[1-9][0-9]\.|[2-9][0-9])' | tail -n 1)
latest_version=14.0.0

if [[ -n "$latest_version" ]]; then
    module load llvm/$latest_version
    echo "Loaded llvm/$latest_version"
else
    echo "No suitable LLVM version found above 13.0.1"
fi
export LLVM_ROOT=$(dirname $(dirname $(which clang)))
export CLANG_OPENCL_INCLUDE=$(dirname $(find $LLVM_ROOT -name "opencl-c.h"))
major_version=$(echo "$latest_version" | awk -F. '{print $1}')

#export LLVM_ROOT=/auto/software/swtree/ubuntu20.04/x86_64/llvm/13.0.1
#confirm we have Oclgrind with AIWC installed
echo "Looking for AIWC in: $AIWC ..."
if [ ! -d "$AIWC" ] 
then
    echo "Not found. Installing to: $AIWC..."
    export WORKING_DIR=`pwd`
    export OCLGRIND_SRC=aiwc-src
    if [ ! -d "$OCLGRIND_SRC" ]; then
        git clone https://github.com/BeauJoh/AIWC.git $OCLGRIND_SRC
        [ $? -ne 0 ] && exit 1
        # Compare the major version to 15
        if [ "$major_version" -gt 15 ]; then
            sed -i -e "s/index\s*==\s*llvm::UndefMaskElem/index != -1/g" $OCLGRIND_SRC/src/core/WorkItem.cpp
            sed -i -e "s/llvm\/Transforms\/IPO\/PassManagerBuilder.h/llvm\/Passes\/PassBuilder.h/g" $OCLGRIND_SRC/src/core/Program.cpp
            sed -i -e "s/startswith/starts_with/g" $OCLGRIND_SRC/src/core/Kernel.cpp
            sed -i -e "s/startswith/starts_with/g" $OCLGRIND_SRC/src/core/Program.cpp
            sed -i -e "s/startswith/starts_with/g" $OCLGRIND_SRC/src/kernel/Simulation.cpp
            for i in core/common.cpp core/Kernel.cpp core/Program.cpp core/WorkItemBuiltins.cpp core/WorkItem.cpp plugins/InstructionCounter.cpp plugins/MemCheck.cpp plugins/Uninitialized.cpp; do 
                sed -i -e "s/getPointerElementType/getNonOpaquePointerElementType/g" $OCLGRIND_SRC/src/$i
            done
            sed -i -e "s/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/g" $OCLGRIND_SRC/CMakeLists.txt
            sed -i -e "s/\${LLVM_VERSION_MAJOR}.\${LLVM_VERSION_MINOR}.\${LLVM_VERSION_PATCH}/\${LLVM_VERSION_MAJOR}/g" $OCLGRIND_SRC/CMakeLists.txt
        fi
    fi
    mkdir -p $OCLGRIND_SRC/build ;
    [ $? -ne 0 ] && exit 1
    cd $OCLGRIND_SRC/build ;
    [ $? -ne 0 ] && exit 1
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$AIWC -DBUILD_SHARED_LIBS=ON -DLLVM_ROOT_DIR=$LLVM_ROOT -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++;
    [ $? -ne 0 ] && exit 1
    make -j$(nproc)
    [ $? -ne 0 ] && exit 1
    make -j install
    [ $? -ne 0 ] && exit 1
    echo "done."
    cd $WORKING_DIR
fi
echo "AIWC found in: $AIWC"

echo "Running OpenCL... (OclGrind)"
OCL_ICD_VENDORS=$AIWC/lib/ OCL_ICD_FILENAMES=liboclgrind-rt.so IRIS_ARCHS=opencl ./test37_opencl_icd
exit 0
