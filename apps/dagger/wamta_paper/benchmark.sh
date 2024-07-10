#!/bin/bash
#this script tests DAGGER support for --local-sizes="ijk:256 256"
export PAYLOAD_SIZE=${PAYLOAD_SIZE:=2048}
export SKIP_SETUP=${SKIP_SETUP:=0}

export SCRIPT_DIR=`realpath ..`
export WORKING_DIR=`realpath .`
#source /auto/software/iris/setup_system.source
source $SCRIPT_DIR/setup_gpu_backends.sh
#export OCL_ICD_VENDORS=/opt/rocm/lib/libOpenCL.so
#export OCL_ICD_FILENAMES=libOpenCL.so
#export IRIS_ARCHS=opencl
#TODO: when evaluating the crc add the kernel.cl and eth_crc32_lut.h as KERNEL targets to copy
export IRIS_ARCHS=$BACKENDS
export IRIS_HISTORY=1

#installed with:
#micromamba create -f dagger.yaml
#micromamba activate dagger
#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: mamba activate dagger)"
  echo "Aborting..."
  exit
fi

#install iris
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
  IRIS_INSTALL_ROOT="$HOME/.iris"
fi
if [[ $SKIP_SETUP -eq 0 ]]; then
  export IRIS_SRC_DIR=`realpath ../../..`
  rm -f kernel.ptx kernel.hip kernel.openmp.so
  rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
  cd $IRIS_SRC_DIR ; ./build.sh; [ $? -ne 0 ] && exit ;
  echo "target kernels are : " $KERNELS
  cd $SCRIPT_DIR ; make clean; make clean-results; make $KERNELS dagger_runner ;
  [ $? -ne 0 ] && exit ;
  cp $KERNELS $WORKING_DIR ; [ $? -ne 0 ] && exit ; cd $WORKING_DIR ;
fi
source $IRIS_INSTALL_ROOT/setup.source

if [[ ! -n "$SKIP_DAG_REGEN" ]]; then
  #generate dagger payload for this experiment
  $SCRIPT_DIR/dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=40 --num-tasks=240 --min-width=6 --max-width=6 --concurrent-kernels="ijk:6" --skips=3 --sandwich --use-data-memory --local-sizes="ijk:8 8"
  [ $? -ne 0 ] && echo "Failed to generate DAG" && exit 1
  mv $SCRIPT_DIR/graph.json $WORKING_DIR/graph.json
fi

#run the payload (note, we have to omit the data policy since it is incompatible with dmem)
for POLICY in roundrobin depend profile random ftf sdq #data
do
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPT_DIR $SCRIPT_DIR/dagger_runner --graph="$WORKING_DIR/graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=40 --num-tasks=240 --min-width=6 --max-width=6 --concurrent-kernels="ijk:6" --skips=3  --sandwich --use-data-memory #--local-sizes="ijk:256 1 1"
  [ $? -ne 0 ] && echo "Failed to run DAG" && exit 1
  mv $SCRIPT_DIR/dagger_runner-$HOST*\.csv $WORKING_DIR/$HOST-$POLICY-time.csv
  #joint plot
  python $SCRIPT_DIR/gantt/gantt.py --dag=$WORKING_DIR/graph.json --timeline=$WORKING_DIR/$HOST-$POLICY-time.csv --timeline-out=$WORKING_DIR/$HOST-$POLICY-timeline.pdf --dag-out=$WORKING_DIR/$HOST-$POLICY-recoloured_dag.pdf  --combined-out=$WORKING_DIR/$HOST-$POLICY-combined.pdf --no-show-kernel-legend --no-show-task-legend #--drop="Internal-*"
done

