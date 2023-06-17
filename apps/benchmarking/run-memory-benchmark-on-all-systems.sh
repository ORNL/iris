#!/bin/bash

export WORKING_DIRECTORY=`pwd`
#Nvidia GPUs
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-cuda.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-opencl.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-openmp.sh"

ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-cuda.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-opencl.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-openmp.sh"

#note leconte has a POWERPC and it doesn't ship with a working OpenCL runtime
ssh -l 9bj leconte  "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-cuda.sh"
ssh -l 9bj leconte  "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-openmp.sh"

#AMD GPUs
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-hip.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-opencl.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-openmp.sh"

ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-hip.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-opencl.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-openmp.sh"

ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-cuda.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-hip.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-opencl.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./memory-scripts/run-membench-openmp.sh"

mkdir -p memory-results && mv membench-*.csv memory-results
