#!/bin/bash

#NOTE: remove the following after @Narasinga adds logic to discard IRIS platforms if the kernel files can't be found or built
export IRIS_ARCHS=opencl
export LD_LIBRARY_PATH=$IRIS/lib:$LD_LIBRARY_PATH

bash ./clean.sh
rm -rf ./build
mkdir build
cd build
cmake ..
make --ignore-errors
echo "Running OpenCL version of the tests..."
IRIS_ARCHS=opencl make test
echo "Done."
echo "Running OpenMP version of the tests..."
IRIS_ARCHS=openmp make test
echo "Done."
echo "Running CUDA version of the tests..."
IRIS_ARCHS=cuda make test
echo "Done."
echo "Running HIP version of the tests..."
IRIS_ARCHS=hip make test
echo "Done."
echo "Running All version of the tests..."
IRIS_ARCHS=openmp,cuda,hip,opencl make test
echo "Done."

