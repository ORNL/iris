#!/bin/bash

#NOTE: remove the following after @Narasinga adds logic to discard IRIS platforms if the kernel files can't be found or built
export IRIS_ARCHS=opencl
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH 

bash ./clean.sh
rm -rf ./build
mkdir build
cd build
cmake ..
make --ignore-errors
make test

