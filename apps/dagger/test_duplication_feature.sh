#!/bin/bash
#uncomment to use data memory rather than explicit memory transfers
export SKIP_SETUP=${SKIP_SETUP:=0}
set -x;
if [ "x$SKIP_SETUP" = "x0" ]; then
source ./setup.sh
fi

#source ./build.sh
[ $? -ne 0 ] &&  exit 1
#uncomment to use explicit (non DMEM) version
#export USE_DATA_MEMORY=" "
echo "using data memory?" $USE_DATA_MEMORY

#data policy is unsupported if DMEM is used
if [ -n "$USE_DATA_MEMORY" ]; then
  export POLICIES=(roundrobin depend profile random ftf sdq);
else
  export POLICIES=(roundrobin depend profile random ftf sdq data);
fi

export SIZES=("10")
export PAYLOAD_SIZE=256
#export SIZES=("10" "25" "100")
export REPEATS=0

echo Using policies: ${POLICIES[@]}
echo Using sizes: ${SIZES[@]}
echo Repeating the experiment $REPEATS times.

mkdir -p dagger-payloads
mkdir -p dagger-results

echo "Running DAGGER on payloads..."
for SIZE in ${SIZES[@]}
do
  echo "*******************************************************************"
  echo "*                          Lineartwo $SIZE                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --graph="dagger-payloads/lineartwo$SIZE-graph.json" --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=1 --sandwich $USE_DATA_MEMORY --duplicates="2"
  [ $? -ne 0 ] && echo "Failed to generate Lineartwo $SIZE" &&  exit 1

  for POLICY in ${POLICIES[@]}
  do
    for (( num_run=0; num_run<=$REPEATS; num_run++ ))
    do
      echo "Running IRIS on Lineartwo $SIZE with Policy: $POLICY  run no. $num_run"
      ./dagger_runner --graph="dagger-payloads/lineartwo$SIZE-graph.json" --logfile="dagger-results/lineartwo${POLICY}${SIZE}${num_run}time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" $USE_DATA_MEMORY --duplicates="2"
      [ $? -ne 0 ] && echo "Lineartwo $SIZE Failed with Policy: $POLICY" &&  exit 1
    done
    #plot timeline with gantt
    if [ "$SIZE" == "10" ] ; then
      python ./gantt/gantt.py --dag="./lineartwo$SIZE-graph.json" --timeline="dagger-results/lineartwo${POLICY}${SIZE}${num_run}time.csv" --combined-out="./lineartwo$SIZE-$POLICY-$SYSTEM.pdf" --no-show-kernel-legend #--keep-memory-transfer-commands # --drop="Initialize-0,Initialize-1" #--title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"
    fi
    [ $? -ne 0 ] && echo "Failed Combined Plotting of Lineartwo $SIZE with Policy: $POLICY" &&  exit 1
  done
done

mkdir -p dagger-graphs; mv lineartwo*.pdf dagger-graphs

echo "Running DAGGER on with concurrency and duplicate payloads..."
for SIZE in ${SIZES[@]}
do
  echo "*******************************************************************"
  echo "*                          Linearthree $SIZE                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --graph="dagger-payloads/linearthree$SIZE-graph.json" --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=1 --sandwich $USE_DATA_MEMORY --duplicates="3" --concurrent-kernels="ijk:5"
  [ $? -ne 0 ] && echo "Failed to generate Linearthree $SIZE" &&  exit 1

  for POLICY in ${POLICIES[@]}
  do
    for (( num_run=0; num_run<=$REPEATS; num_run++ ))
    do
      echo "Running IRIS on Linearthree $SIZE with Policy: $POLICY  run no. $num_run"
      ./dagger_runner --graph="dagger-payloads/linearthree$SIZE-graph.json" --logfile="dagger-results/linearthree${POLICY}${SIZE}${num_run}time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" $USE_DATA_MEMORY --duplicates="3" --concurrent-kernels="ijk:5"
      [ $? -ne 0 ] && echo "Linearthree $SIZE Failed with Policy: $POLICY" &&  exit 1
    done
    #plot timeline with gantt
    if [ "$SIZE" == "10" ] ; then
      python ./gantt/gantt.py --dag="./linearthree$SIZE-graph.json" --timeline="dagger-results/linearthree${POLICY}${SIZE}${num_run}time.csv" --combined-out="./linearthree$SIZE-$POLICY-$SYSTEM.pdf" --no-show-kernel-legend #--keep-memory-transfer-commands # --drop="Initialize-0,Initialize-1" #--title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"
    fi
    [ $? -ne 0 ] && echo "Failed Combined Plotting of Linearthree $SIZE with Policy: $POLICY" &&  exit 1
  done
done

mkdir -p dagger-graphs; mv linearthree*.pdf dagger-graphs

