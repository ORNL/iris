#!/bin/bash

export WORKING_DIRECTORY=`pwd`
#Nvidia GPUs
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-cuda.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-opencl.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-openmp.sh"

ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-cuda.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-opencl.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-openmp.sh"

#note leconte has a POWERPC and it doesn't ship with a working OpenCL runtime
ssh -l 9bj leconte  "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-cuda.sh"
ssh -l 9bj leconte  "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-openmp.sh"

#AMD GPUs
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-hip.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-opencl.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-openmp.sh"

ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-hip.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-opencl.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-openmp.sh"

ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-cuda.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-hip.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-opencl.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./compute-performance-scripts/run-dgemm-openmp.sh"

mkdir -p compute-performance-results && mv dgemm-*.csv compute-performance-results
