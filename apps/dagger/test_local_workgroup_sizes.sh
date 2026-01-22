#!/bin/bash
#this script tests DAGGER support for --local-sizes="ijk:256 256"
export PAYLOAD_SIZE=${PAYLOAD_SIZE:=1024}
export SKIP_SETUP=${SKIP_SETUP:=0}
export SCRIPT_DIR=`realpath .`
export WORKING_DIR=`realpath .`
set -x
#source $SCRIPT_DIR/setup_backends.sh
#export IRIS_ARCHS=$BACKENDS
# export IRIS_ARCHS=opencl
#$BACKENDS
export IRIS_HISTORY=1

if [[ "$IRIS_ARCHS" == *"opencl"* ]]; then
  echo using smaller set of workgroups with OpenCL since AMD drivers were misconfigured for max_work_group_size
  export LOCAL_SIZES=("1 1" "2 1" "4 1" "8 1" "16 1" "32 1" "64 1" "128 1" "256 1" \
  "1 1" "1 2" "1 4" "1 8" "1 16" "1 32" "1 64" "1 128" "1 256"\
  "1 1" "2 2" "4 4" "8 8" "16 16")
  export DIMS=("x" "x" "x" "x" "x" "x" "x" "x" "x" \
  "y" "y" "y" "y" "y" "y" "y" "y" "y" \
  "xy" "xy" "xy" "xy" "xy")
else
  export LOCAL_SIZES=("1 1" "2 1" "4 1" "8 1" "16 1" "32 1" "64 1" "128 1" "256 1" "512 1" "1024 1" \
  "1 1" "1 2" "1 4" "1 8" "1 16" "1 32" "1 64" "1 128" "1 256" "1 512" "1 1024"\
  "1 1" "2 2" "4 4" "8 8" "16 16" "32 32")
  export DIMS=("x" "x" "x" "x" "x" "x" "x" "x" "x" "x" "x" \
  "y" "y" "y" "y" "y" "y" "y" "y" "y" "y" "y"\
  "xy" "xy" "xy" "xy" "xy" "xy")
fi

#installed with:
#micromamba create -f dagger.yaml
#micromamba activate dagger
#if we don't have a conda env set, then load it.
INVENV=$(python3 -c 'import sys; print ("1" if sys.prefix != sys.base_prefix else "0")')
if [[ -z "$CONDA_PREFIX" ]] && [[ $INVENV == 0 ]] ; then
  echo "Please ensure this script is run from a conda session or python venv (hint: mamba activate dagger)"
  echo "Aborting..."
  exit
fi

#install iris
if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
  IRIS_INSTALL_ROOT="$HOME/.iris"
fi
if [[ $SKIP_SETUP -eq 0 ]]; then
  export IRIS_SRC_DIR=`realpath ../..`
  rm -f kernel.ptx kernel.hip kernel.openmp.so
  rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
  cd $IRIS_SRC_DIR ; ./build.sh; [ $? -ne 0 ] && exit ;
fi
source $IRIS_INSTALL_ROOT/setup.source

echo "target kernels are : " $KERNELS
cd $SCRIPT_DIR ; make clean; make clean-results; make $KERNELS dagger_runner ;
[ $? -ne 0 ] && exit ;
cd $WORKING_DIR ;

echo "Running DAGGER on payloads..."
rm -f dagger-results/lws_times.csv
mkdir -p dagger-results
mkdir -p dagger-graphs
touch dagger-results/lws_times.csv
echo "size,secs,dim" > dagger-results/lws_times.csv
for ((idx=0; idx<${#LOCAL_SIZES[@]}; idx++)); do
  export LWS="${LOCAL_SIZES[idx]}"
  export DIM="${DIMS[idx]}"
  echo "*******************************************************************"
  echo "Generating DAG (using --local-sizes=ijk:$LWS"
  echo "*******************************************************************"
  # generate dagger payload for this experiment
  $SCRIPT_DIR/dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=40 --num-tasks=240 --min-width=6 --max-width=6 --concurrent-kernels="ijk:6" --skips=3 --sandwich $USE_DATA_MEMORY --local-sizes="ijk:$LWS"
  [ $? -ne 0 ] && echo "Failed to generate DAG" && exit 1
  export FILENAME=${LWS// /x}
  mkdir -p dagger-payloads
  mv $SCRIPT_DIR/graph.json $WORKING_DIR/dagger-payloads/lws-graph-$FILENAME.json
  #run the payload (note, we have to omit the data policy since it is incompatible with dmem)
  for POLICY in roundrobin #depend profile random ftf sdq
  do
    start=`date +%s.%N`
    export IRIS_HISTORY_FILE=$WORKING_DIR/dagger-results/$HOST-$POLICY-lws-$FILENAME-time.csv
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPT_DIR $SCRIPT_DIR/dagger_runner --graph="$WORKING_DIR/dagger-payloads/lws-graph-$FILENAME.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=40 --num-tasks=240 --min-width=6 --max-width=6 --concurrent-kernels="ijk:6" --skips=3  --sandwich $USE_DATA_MEMORY
    [ $? -ne 0 ] && echo "Failed to run DAG" && exit 1
    end=`date +%s.%N`
    runtime=$( echo "$end - $start" | bc -l )
    echo "RUNTIME took $runtime"
    echo "$FILENAME,$runtime,$DIM" >> dagger-results/lws_times.csv
    #joint plot
    python $SCRIPT_DIR/gantt/gantt.py --dag="$WORKING_DIR/dagger-payloads/lws-graph-$FILENAME.json" --timeline="$WORKING_DIR/dagger-results/$HOST-$POLICY-lws-$FILENAME-time.csv" --timeline-out="$WORKING_DIR/dagger-graphs/$HOST-$POLICY-time-lws-graph-$FILENAME-timeline.pdf" --dag-out="$WORKING_DIR/dagger-graphs/$HOST-$POLICY-time-lws-graph-$FILENAME-recoloured_dag.pdf"  --combined-out="$WORKING_DIR/dagger-graphs/$HOST-$POLICY-time-lws-graph-$FILENAME-combined.pdf" --no-show-kernel-legend --no-show-task-legend #--drop="Internal-*"
  done

done
python ./plot_local_workgroup_sizes.py
([ $? -ne 0 ] && echo "Failed plot the combined timing results" && exit 1) || true

