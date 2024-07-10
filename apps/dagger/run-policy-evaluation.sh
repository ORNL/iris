#!/bin/bash

source ./build.sh
[ $? -ne 0 ] &&  exit 1

#data policy is unsupported if DMEM is used
if [ -n "$USE_DATA_MEMORY" ]; then
  export POLICIES=(roundrobin depend profile random ftf sdq);
else
  export POLICIES=(roundrobin depend profile random ftf sdq data);
fi
export SIZES=("10" "25" "100")

echo Using policies: ${POLICIES[@]}
echo Using sizes: ${SIZES[@]}

export RESULTS_DIR=`pwd`/dagger-results
export GRAPHS_DIR=`pwd`/dagger-graphs
mkdir -p $RESULTS_DIR $GRAPHS_DIR
echo "Running DAGGER evaluation.... (graph figures can be found in $GRAPHS_DIR)"

#ensure libiris.so is in the shared library path
#  echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH

#Only run DAGGER once to generate the payloads to test the systems (we want to compare the scheduling algorithms over different systems, and so we should fix the payloads over the whole experiment)
#remove the dagger-payloads directory to regenerate payloads
if ! [ -d dagger-payloads ] ; then
  echo "Generating DAGGER payloads (delete this directory to regenerate new DAG payloads)..."
  mkdir -p dagger-payloads
  echo "*******************************************************************"
  echo "*                          Linear 10                              *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=8 --num-tasks=8 --min-width=1 --max-width=1 --sandwich $USE_DATA_MEMORY
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/linear10-graph.json
  cp dag.pdf $GRAPHS_DIR/linear10-graph.pdf
  echo "*******************************************************************"
  echo "*                          Linear 25                              *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=1 $USE_DATA_MEMORY

  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/linear25-graph.json
  cp dag.pdf $GRAPHS_DIR/linear25-graph.pdf
  echo "*******************************************************************"
  echo "*                          Linear 100                             *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1 $USE_DATA_MEMORY
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/linear100-graph.json
  cp dag.pdf $GRAPHS_DIR/linear100-graph.pdf
  #echo "*******************************************************************"
  #echo "*                          Parallel 2by10                         *"
  #echo "*******************************************************************"
  #./dagger_generator.py --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 $USE_DATA_MEMORY
  #[ $? -ne 0 ] &&  exit 1
  #cat graph.json
  #cp graph.json dagger-payloads/parallel2by10-graph.json
  #cp dag.pdf $GRAPHS_DIR/parallel2by10-graph.pdf
  #echo "*******************************************************************"
  #echo "*                          Parallel 5by100                        *"
  #echo "*******************************************************************"
  #./dagger_generator.py --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1 $USE_DATA_MEMORY
  #[ $? -ne 0 ] &&  exit 1
  #cat graph.json
  #cp graph.json dagger-payloads/parallel5by100-graph.json
  #cp dag.pdf $GRAPHS_DIR/parallel5by100-graph.pdf
  echo "*******************************************************************"
  echo "*                          Diamond 10                             *"
  echo "*******************************************************************"
  #diamond 10
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=8 --min-width=8 --max-width=8 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich  $USE_DATA_MEMORY --concurrent-kernels="ijk:8"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond10-graph.json
  cp dag.pdf $GRAPHS_DIR/diamond10-graph.pdf
  echo "*******************************************************************"
  echo "*                          Diamond 25                             *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=23 --min-width=23 --max-width=23 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:23"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond25-graph.json
  cp dag.pdf $GRAPHS_DIR/diamond25-graph.pdf
  echo "*******************************************************************"
  echo "*                          Diamond 100                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=98 --min-width=98 --max-width=98 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:98"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond100-graph.json
  cp dag.pdf $GRAPHS_DIR/diamond100-graph.pdf
  echo "*******************************************************************"
  echo "*                          Diamond 1000                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=998 --min-width=998 --max-width=998 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:998"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond1000-graph.json
  cp dag.pdf $GRAPHS_DIR/diamond1000-graph.pdf
  echo "*******************************************************************"
  echo "*                          Chainlink 10                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=5 --num-tasks=8 --min-width=1 --max-width=2 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --cdf-mean=1.5 --cdf-std-dev=0 --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:2" 
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/chainlink10-graph.json
  cp dag.pdf $GRAPHS_DIR/chainlink10-graph.pdf
  echo "*******************************************************************"
  echo "*                          Chainlink 25                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=12 --num-tasks=25 --min-width=1 --max-width=2 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=1.5 --cdf-std-dev=0 $USE_DATA_MEMORY --concurrent-kernels="ijk:2"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/chainlink25-graph.json
  cp dag.pdf $GRAPHS_DIR/chainlink25-graph.pdf
  echo "*******************************************************************"
  echo "*                          Chainlink 100                          *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=50 --num-tasks=100 --min-width=1 --max-width=2 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=1.5 --cdf-std-dev=0 $USE_DATA_MEMORY --concurrent-kernels="ijk:2"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/chainlink100-graph.json
  cp dag.pdf $GRAPHS_DIR/chainlink100-graph.pdf
  echo "*******************************************************************"
  echo "*                          Galaga 25                              *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 $USE_DATA_MEMORY --concurrent-kernels="ijk:12"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/galaga25-graph.json
  cp dag.pdf $GRAPHS_DIR/galaga25-graph.pdf
  echo "*******************************************************************"
  echo "*                          Tangled 10                             *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=12 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3 $USE_DATA_MEMORY --concurrent-kernels="ijk:12"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/tangled10-graph.json
  cp dag.pdf $GRAPHS_DIR/tangled10-graph.pdf
  echo "*******************************************************************"
  echo "*                          Tangled 25                             *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3 $USE_DATA_MEMORY --concurrent-kernels="ijk:12"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/tangled25-graph.json
  cp dag.pdf $GRAPHS_DIR/tangled25-graph.pdf
  echo "*******************************************************************"
  echo "*                          Tangled 100                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=12 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3 $USE_DATA_MEMORY --concurrent-kernels="ijk:12"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/tangled100-graph.json
  cp dag.pdf $GRAPHS_DIR/tangled100-graph.pdf
  echo "*******************************************************************"
  echo "*                           Brain 1000                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10 $USE_DATA_MEMORY --concurrent-kernels="ijk:50"
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/brain1000-graph.json
  cp dag.pdf $GRAPHS_DIR/brain1000-graph.pdf
fi

echo "Running deep-dive test..."
echo "Running IRIS on Linear 10 with Policy: roundrobin"
./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --concurrent-kernels="ijk:1"
[ $? -ne 0 ] &&  exit 1
cat graph.json
cp graph.json dagger-payloads/linear10-graph-dmem.json
cp dag.pdf $GRAPHS_DIR/linear10-graph-dmem.pdf
./dagger_runner --graph="dagger-payloads/linear10-graph-dmem.json" --repeats=1 --scheduling-policy="roundrobin" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory
[ $? -ne 0 ] && echo "Linear 10 Failed with Policy: roundrobin" &&  exit 1
#archive result
mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/datamemlinear10-roundrobin-$SYSTEM-0.csv
#plot timeline with gantt
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph-dmem.json --timeline=$RESULTS_DIR/datamemlinear10-roundrobin-$SYSTEM-0.csv --timeline-out=$GRAPHS_DIR/datamemlinear10-roundrobin-$SYSTEM-timeline.pdf --dag-out=$GRAPHS_DIR/datamemlinear10-roundrobin-$SYSTEM-dag.pdf  --combined-out=$GRAPHS_DIR/datamemlinear10-roundrobin-$SYSTEM.pdf --no-show-kernel-legend
[ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 with Policy: roundrobin" &&  exit 1
echo "Passed."

echo "Testing the same with --sandwich argument enabled."
./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --concurrent-kernels="ijk:1" --sandwich
[ $? -ne 0 ] &&  exit 1
cat graph.json
cp graph.json dagger-payloads/linear10-graph-dmem-sandwich.json
cp dag.pdf $GRAPHS_DIR/linear10-graph-dmem-sandwich.pdf
./dagger_runner --graph="dagger-payloads/linear10-graph-dmem-sandwich.json" --repeats=1 --scheduling-policy="roundrobin" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --sandwich
[ $? -ne 0 ] && echo "Linear 10 Failed with Policy: roundrobin" &&  exit 1
#archive result
mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/sandwichdatamemlinear10-roundrobin-$SYSTEM-0.csv
#plot timeline with gantt
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph-dmem-sandwich.json --timeline=$RESULTS_DIR/sandwichdatamemlinear10-roundrobin-$SYSTEM-0.csv --timeline-out=$GRAPHS_DIR/sandwichdatamemlinear10-roundrobin-$SYSTEM-timeline.pdf --dag-out=$GRAPHS_DIR/sandwichdatamemlinear10-roundrobin-$SYSTEM-dag.pdf  --combined-out=$GRAPHS_DIR/sandwichdatamemlinear10-roundrobin-$SYSTEM.pdf --no-show-kernel-legend
[ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 with Policy: roundrobin DMEM and sandwich" &&  exit 1
echo "Passed."

echo "Running memory-shuffle test..."
echo "Running IRIS (explicit memory-shuffle) on Linear 10 with Policy: roundrobin"
./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --num-memory-shuffles=5 --concurrent-kernels="ijk:3"
[ $? -ne 0 ] &&  exit 1
cat graph.json
cp graph.json dagger-payloads/linear10-graph-memshuf.json
cp dag.pdf $GRAPHS_DIR/linear10-graph-memshuf.pdf
./dagger_runner --graph="dagger-payloads/linear10-graph-memshuf.json" --repeats=1 --scheduling-policy="roundrobin" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --concurrent-kernels="ijk:3"
[ $? -ne 0 ] && echo "Linear 10 (explicit memory-shuffle) Failed with Policy: roundrobin" &&  exit 1
#archive result
mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/memshuflinear10-roundrobin-$SYSTEM-0.csv
#plot timeline with gantt
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph-memshuf.json --timeline=$RESULTS_DIR/memshuflinear10-roundrobin-$SYSTEM-0.csv --timeline-out=$GRAPHS_DIR/memshuflinear10-roundrobin-$SYSTEM-timeline.pdf --dag-out=$GRAPHS_DIR/memshuflinear10-roundrobin-$SYSTEM-dag.pdf  --combined-out=$GRAPHS_DIR/memshuflinear10-roundrobin-$SYSTEM.pdf --no-show-kernel-legend
[ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 (explicit memory-shuffle) with Policy: roundrobin" &&  exit 1
echo "Passed."

echo "Running IRIS (memory-shuffle with DMEM) on Linear 10 with Policy: roundrobin"
./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --num-memory-shuffles=5 --concurrent-kernels="ijk:3"
[ $? -ne 0 ] &&  exit 1
cat graph.json
cp graph.json dagger-payloads/linear10-graph-dmemshuf.json
cp dag.pdf $GRAPHS_DIR/linear10-graph-dmemshuf.pdf
./dagger_runner --graph="dagger-payloads/linear10-graph-dmemshuf.json" --repeats=1 --scheduling-policy="roundrobin" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --concurrent-kernels="ijk:3"
[ $? -ne 0 ] && echo "Linear 10 (memory-shuffle with DMEM) Failed with Policy: roundrobin" &&  exit 1
#archive result
mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/dmemshuflinear10-roundrobin-$SYSTEM-0.csv
#plot timeline with gantt
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph-dmemshuf.json --timeline=$RESULTS_DIR/dmemshuflinear10-roundrobin-$SYSTEM-0.csv --timeline-out=$GRAPHS_DIR/dmemshuflinear10-roundrobin-$SYSTEM-timeline.pdf --dag-out=$GRAPHS_DIR/dmemshuflinear10-roundrobin-$SYSTEM-dag.pdf  --combined-out=$GRAPHS_DIR/dmemshuflinear10-roundrobin-$SYSTEM.pdf --no-show-kernel-legend
[ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 (memory-shuffle with DMEM) with Policy: roundrobin" &&  exit 1
echo "Passed."


echo "Running memory-shuffle (handover) test..."
echo "Running IRIS (explicit memory-shuffle with handover) on Linear 10 with Policy: roundrobin"
./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --num-memory-shuffles=5 --concurrent-kernels="ijk:3"  --handover-in-memory-shuffle
[ $? -ne 0 ] &&  exit 1
cat graph.json
cp graph.json dagger-payloads/linear10-graph-memshuf-handover.json
cp dag.pdf $GRAPHS_DIR/linear10-graph-memshuf-handover.pdf
./dagger_runner --graph="dagger-payloads/linear10-graph-memshuf-handover.json"  --repeats=1 --scheduling-policy="roundrobin" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --concurrent-kernels="ijk:3"
[ $? -ne 0 ] && echo "Linear 10 (explicit memory-shuffle with handover) Failed with Policy: roundrobin" &&  exit 1
#archive result
mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/memshufhandlinear10-roundrobin-$SYSTEM-0.csv
#plot timeline with gantt
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph-memshuf-handover.json --timeline=$RESULTS_DIR/memshufhandlinear10-roundrobin-$SYSTEM-0.csv --timeline-out=$GRAPHS_DIR/memshufhandlinear10-roundrobin-$SYSTEM-timeline.pdf --dag-out=$GRAPHS_DIR/memshufhandlinear10-roundrobin-$SYSTEM-dag.pdf  --combined-out=$GRAPHS_DIR/memshufhandlinear10-roundrobin-$SYSTEM.pdf --no-show-kernel-legend
[ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 (explicit memory-shuffle with handover) with Policy: roundrobin" &&  exit 1
echo "Passed."

echo "Running IRIS (memory-shuffle with DMEM and handover) on Linear 10 with Policy: roundrobin"
./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --num-memory-shuffles=5 --concurrent-kernels="ijk:3" --handover-in-memory-shuffle
[ $? -ne 0 ] &&  exit 1
cat graph.json
cp graph.json dagger-payloads/linear10-graph-dmemshuf-handover.json
cp dag.pdf $GRAPHS_DIR/linear10-graph-dmemshuf-handover.pdf
./dagger_runner --graph="dagger-payloads/linear10-graph-dmemshuf-handover.json"  --repeats=1 --scheduling-policy="roundrobin" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory --concurrent-kernels="ijk:3"
[ $? -ne 0 ] && echo "Linear 10 (memory-shuffle with DMEM and handover) Failed with Policy: roundrobin" &&  exit 1
#archive result
mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/dmemshufhandlinear10-roundrobin-$SYSTEM-0.csv
#plot timeline with gantt
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph-dmemshuf-handover.json --timeline=$RESULTS_DIR/dmemshufhandlinear10-roundrobin-$SYSTEM-0.csv --timeline-out=$GRAPHS_DIR/dmemshufhandlinear10-roundrobin-$SYSTEM-timeline.pdf --dag-out=$GRAPHS_DIR/dmemshufhandlinear10-roundrobin-$SYSTEM-dag.pdf  --combined-out=$GRAPHS_DIR/dmemshufhandlinear10-roundrobin-$SYSTEM.pdf --no-show-kernel-legend
[ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 (memory-shuffle with DMEM and handover) with Policy: roundrobin" &&  exit 1
echo "Passed."


echo "Running DAGGER on payloads..."
for SIZE in ${SIZES[@]}
do
  echo "*******************************************************************"
  echo "*                          Linear $SIZE                           *"
  echo "*******************************************************************"
  cp dagger-payloads/linear$SIZE-graph.json graph.json ; cat graph.json
  for POLICY in ${POLICIES[@]}
  do
    for (( num_run=0; num_run<=$REPEATS; num_run++ ))
    do
      echo "Running IRIS on Linear $SIZE with Policy: $POLICY  run no. $num_run"
      ./dagger_runner --graph="dagger-payloads/linear$SIZE-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=1 $USE_DATA_MEMORY
      [ $? -ne 0 ] && echo "Linear $SIZE Failed with Policy: $POLICY" &&  exit 1
      #archive result
      mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/linear$SIZE-$POLICY-$SYSTEM-$num_run.csv
    done
    #plot timeline with gantt
    if [ "$SIZE" == "10" ] ; then
      python ./gantt/gantt.py --dag=./dagger-payloads/linear$SIZE-graph.json --timeline=$RESULTS_DIR/linear$SIZE-$POLICY-$SYSTEM-0.csv --combined-out=$GRAPHS_DIR/linear$SIZE-$POLICY-$SYSTEM.pdf --no-show-kernel-legend #--keep-memory-transfer-commands # --drop="Initialize-0,Initialize-1" #--title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"
    fi
    [ $? -ne 0 ] && echo "Failed Combined Plotting of Linear $SIZE with Policy: $POLICY" &&  exit 1
  done
done

# Parallel 2-by-10
#echo "*******************************************************************"
#echo "*                          Parallel 2by10                         *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
##[ $? -ne 0 ] &&  exit 1
#cp graph.json dagger-payloads/parallel2by10-graph.json
#for POLICY in ${POLICIES[@]}
#do
#  echo "Running IRIS on Parallel 2by10 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/parallel2by10-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE  --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 $USE_DATA_MEMORY
#  [ $? -ne 0 ] && echo "Parallel 2by10 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/parallel2by10-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done
#
#echo "*******************************************************************"
#echo "*                          Parallel 5by100                        *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/parallel5by100-graph.json
##cp dag.pdf $RESULTS_DIR/parallel5by100-graph.pdf
#cp dagger-payloads/parallel5by100-graph.json graph.json ; cat graph.json
#for POLICY in ${POLICIES[@]}
#do
#  echo "Running IRIS on Parallel 5by100 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="parallel5by100-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE  --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1 $USE_DATA_MEMORY
#  [ $? -ne 0 ] && echo "Parallel 5by100 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/parallel5by100-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

for SIZE in ${SIZES[@]}
do
  echo "*******************************************************************"
  echo "*                          Diamond $SIZE                          *"
  echo "*******************************************************************"
  cp dagger-payloads/diamond$SIZE-graph.json graph.json ; cat graph.json
  for POLICY in ${POLICIES[@]}
  do
    for (( num_run=0; num_run<=$REPEATS; num_run++ ))
    do
      echo "Running IRIS on Diamond $SIZE with Policy: $POLICY  run no. $num_run"
      IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/diamond$SIZE-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=$SIZE --min-width=$SIZE --max-width=$SIZE --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:$(($SIZE-2))"
      [ $? -ne 0 ] && echo "Diamond $SIZE Failed with Policy: $POLICY" &&  exit 1
      mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/diamond$SIZE-$POLICY-$SYSTEM-$num_run.csv
      [ $? -ne 0 ] &&  exit 1
    done
    #plot timeline with gantt
    if [ "$SIZE" == "10" ] ; then
      python ./gantt/gantt.py --dag=./dagger-payloads/diamond$SIZE-graph.json --timeline=$RESULTS_DIR/diamond$SIZE-$POLICY-$SYSTEM-0.csv --combined-out=$GRAPHS_DIR/diamond$SIZE-$POLICY-$SYSTEM.pdf
      [ $? -ne 0 ] && echo "Failed Combined Plotting of Diamond $SIZE with Policy: $POLICY" &&  exit 1
    fi
  done
done
#module load nvhpc/23.7
for SIZE in ${SIZES[@]}
do
  echo "*******************************************************************"
  echo "*                          Chainlink $SIZE                        *"
  echo "*******************************************************************"
  cp dagger-payloads/chainlink$SIZE-graph.json graph.json ; cat graph.json
  for POLICY in ${POLICIES[@]}
  do
    for (( num_run=0; num_run<=$REPEATS; num_run++ ))
    do
      echo "Running IRIS on Chainlink $SIZE with Policy: $POLICY  run no. $num_run"
      ./dagger_runner --graph="dagger-payloads/chainlink$SIZE-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=2  --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:2"
      [ $? -ne 0 ] && echo "Chainlink $SIZE Failed with Policy: $POLICY at Run no. $num_run and with Size: $SIZE and with $USE_DATA_MEMORY" &&  exit 1
      mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/chainlink$SIZE-$POLICY-$SYSTEM-$num_run.csv
      [ $? -ne 0 ] &&  exit 1
    done
    #plot timeline with gantt
    if [ "$SIZE" == "10" ] ; then
      python ./gantt/gantt.py --dag=./dagger-payloads/chainlink$SIZE-graph.json --timeline=$RESULTS_DIR/chainlink$SIZE-$POLICY-$SYSTEM-0.csv --combined-out=$GRAPHS_DIR/chainlink$SIZE-$POLICY-$SYSTEM.pdf
      [ $? -ne 0 ] && echo "Failed Combined Plotting of Chainlink $SIZE with Policy: $POLICY" &&  exit 1
    fi
  done
done

#echo "*******************************************************************"
#echo "*                          Galaga 25                              *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/galaga-25-graph.json
##cp dag.pdf $RESULTS_DIR/galaga-25-graph.pdf
#cp dagger-payloads/galaga-25-graph.json graph.json ; cat graph.json
#for POLICY in ${POLICIES[@]}
#do
#  echo "Running IRIS on Galaga 25 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/galaga-25-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich $USE_DATA_MEMORY
#  [ $? -ne 0 ] && echo "Galaga 25 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/chainlink-25-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done
for SIZE in ${SIZES[@]}
do
  echo "*******************************************************************"
  echo "*                          Tangled $SIZE                             *"
  echo "*******************************************************************"
  cp dagger-payloads/tangled$SIZE-graph.json graph.json ; cat graph.json
  for POLICY in ${POLICIES[@]}
  do
    for (( num_run=0; num_run<=$REPEATS; num_run++ ))
    do
      echo "Running IRIS on Tangled $SIZE with Policy: $POLICY  run no. $num_run"
      IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/tangled$SIZE-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=12  --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:12"
      [ $? -ne 0 ] && echo "Tangled $SIZE Failed with Policy: $POLICY" &&  exit 1
      mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/tangled$SIZE-$POLICY-$SYSTEM-$num_run.csv
      [ $? -ne 0 ] &&  exit 1
    done
    #plot timeline with gantt
    if [ "$SIZE" == "10" ] ; then
      python ./gantt/gantt.py --dag=./dagger-payloads/tangled$SIZE-graph.json --timeline=$RESULTS_DIR/tangled$SIZE-$POLICY-$SYSTEM-0.csv --combined-out=$GRAPHS_DIR/tangled$SIZE-$POLICY-$SYSTEM.pdf
      [ $? -ne 0 ] && echo "Failed Combined Plotting of Tangled $SIZE with Policy: $POLICY" &&  exit 1
    fi
  done
done

#echo "*******************************************************************"
#echo "*                           Brain 1000                            *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/brain-1000-graph.json
##cp dag.pdf $RESULTS_DIR/brain-100-graph.pdf
#cp dagger-payloads/brain-1000-graph.json graph.json ; cat graph.json
#for POLICY in ${POLICIES[@]}
#do
#  echo "Running IRIS on Brain 1000 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/brain-1000-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --sandwich $USE_DATA_MEMORY
#  [ $? -ne 0 ] && echo "Brain 1000 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/brain-1000-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

#TODO: add a mixed kernels test
#save
#rm -rf linear-10-results; mkdir -p linear-10-results; mv linear-10-*.csv linear-10-results
#echo "All results logged into ./linear-10-results"
exit 0
