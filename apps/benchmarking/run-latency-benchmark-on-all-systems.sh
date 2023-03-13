#!/bin/bash

export WORKING_DIRECTORY=`pwd`
#Nvidia GPUs
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-cuda.sh"
ssh -l 9bj oswald00 "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-opencl.sh"

ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-cuda.sh"
ssh -l 9bj equinox  "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-opencl.sh"

#note leconte has a POWE$WORKING_DIRECTORY 
ssh -l 9bj leconte  "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-cuda.sh"

#AMD GPUs
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY &&  ./latency-scripts/run-kernel-launch-overhead-hip.sh"
ssh -l 9bj radeon   "cd $WORKING_DIRECTORY &&  ./latency-scripts/run-kernel-launch-overhead-opencl.sh"

ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-hip.sh"
ssh -l 9bj explorer "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-opencl.sh"

ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-cuda.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-hip.sh"
ssh -l 9bj zenith   "cd $WORKING_DIRECTORY && ./latency-scripts/run-kernel-launch-overhead-opencl.sh"

mkdir -p latency-results && mv kernellaunch-*.csv latency-results
