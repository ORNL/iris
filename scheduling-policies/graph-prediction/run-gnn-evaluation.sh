#!/bin/bash

export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export WORKING_DIRECTORY=`pwd`
export PAYLOAD_SIZE=4096
export SIZES=(10)
#export SIZES=(10 25 100)
export REPEATS=10

#TODO remove when working
cd ../..
source ./build.sh
[ $? -ne 0 ] &&  exit 1

cd $WORKING_DIRECTORY
cd dagger
export SKIP_SETUP=1
source ./build.sh
[ $? -ne 0 ] &&  exit 1

cp $WORKING_DIRECTORY/PolicyGNN.cpp .
make libPolicyGNN.so
[ $? -ne 0 ] &&  exit 1
#does setting the LD_LIBRARY_PATH allow custom policies to be found?
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH

export POLICIES=(custom:gnn:libPolicyGNN.so); #succeed
#export POLICIES=(custom:gnn); #fail
#export POLICIES=(custom:libPolicyGNN.so); #fail
#export POLICIES=(custom); #fail

echo Using policies: ${POLICIES[@]}
echo Using sizes: ${SIZES[@]}

export RESULTS_DIR=$WORKING_DIRECTORY/results
export GRAPHS_DIR=$WORKING_DIRECTORY/graphs
mkdir -p $RESULTS_DIR $GRAPHS_DIR
echo "Running DAGGER evaluation on GNN policy.... (graph figures can be found in $GRAPHS_DIR)"

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
      gdb --args ./dagger_runner --graph="dagger-payloads/linear$SIZE-graph.json" --repeats=1 --scheduling-policy="$POLICY" --size=$PAYLOAD_SIZE  --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=1 $USE_DATA_MEMORY
      exit
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
