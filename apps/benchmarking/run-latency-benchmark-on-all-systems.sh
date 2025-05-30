#!/bin/bash

export WORKING_DIRECTORY=`pwd`
#Nvidia GPUs
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-cuda.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-opencl.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-openmp.sh"

ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-cuda.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-opencl.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-openmp.sh"
#
##note leconte has a POWE$WORKING_DIRECTORY 
#ssh -l 9bj leconte  "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-cuda.sh"
#
#AMD GPUs
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY &&  .task-/latency-scripts/run-kernel-launch-overhead-hip.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY &&  .task-/latency-scripts/run-kernel-launch-overhead-opencl.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY &&  .task-/latency-scripts/run-kernel-launch-overhead-openmp.sh"

ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-hip.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-opencl.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-openmp.sh"

ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-cuda.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-hip.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-opencl.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./task-latency-scripts/run-kernel-launch-overhead-openmp.sh"

mkdir -p task-latency-results && mv kernellaunch-*.csv task-latency-results
