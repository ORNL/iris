#!/bin/bash
#source /etc/profile
#source /etc/profile.d/z00_lmod.sh
#source ~/.bashrc
#source ./setup.sh

if [ "$SYSTEM" = "leconte" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "equinox" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip
elif [ "$SYSTEM" = "radeon" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip
elif [ "$SYSTEM" = "zenith" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.ptx
elif [ "$SYSTEM" = "orc-open-hyp" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.ptx
else
  echo "Unknown system." && exit 1
fi

# exit 1 if the last program run wasn't successful
[ $? -ne 0 ] &&  exit 1

#don't proceed if the target failed to build
if ! [ -f dagger_runner ] ; then
   echo "No dagger_runner app! " && exit 1
fi

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
  ./dagger_generator.py --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/linear10-graph.json
  cp dag.png $GRAPHS_DIR/linear10-graph.png
  echo "*******************************************************************"
  echo "*                          Parallel 2by10                         *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/parallel-2by10-graph.json
  cp dag.png $GRAPHS_DIR/parallel-2by10-graph.png
  echo "*******************************************************************"
  echo "*                          Parallel 5by100                        *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/parallel-5by100-graph.json
  cp dag.png $GRAPHS_DIR/parallel-5by100-graph.png
  echo "*******************************************************************"
  echo "*                          Diamond 10                             *"
  echo "*******************************************************************"
  #diamond 10
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond-10-graph.json
  cp dag.png $GRAPHS_DIR/diamond-10-graph.png
  echo "*******************************************************************"
  echo "*                          Diamond 100                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=100 --min-width=100 --max-width=100 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond-100-graph.json
  cp dag.png $GRAPHS_DIR/diamond-100-graph.png
  echo "*******************************************************************"
  echo "*                          Diamond 1000                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/diamond-1000-graph.json
  cp dag.png $GRAPHS_DIR/diamond-1000-graph.png
  echo "*******************************************************************"
  echo "*                          Chainlink 25                           *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/chainlink-25-graph.json
  cp dag.png $GRAPHS_DIR/chainlink-25-graph.png
  echo "*******************************************************************"
  echo "*                          Galaga 25                              *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/galaga-25-graph.json
  cp dag.png $GRAPHS_DIR/galaga-25-graph.png
  echo "*******************************************************************"
  echo "*                          Tangled 25                             *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/tangled-25-graph.json
  cp dag.png $GRAPHS_DIR/tangled-25-graph.png
  echo "*******************************************************************"
  echo "*                           Brain 1000                            *"
  echo "*******************************************************************"
  ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10
  [ $? -ne 0 ] &&  exit 1
  cat graph.json
  cp graph.json dagger-payloads/brain-1000-graph.json
  cp dag.png $GRAPHS_DIR/brain-1000-graph.png
fi

echo "Running DAGGER on payloads..."
echo "*******************************************************************"
echo "*                          Linear 10                              *"
echo "*******************************************************************"
##build linear-10 DAG
#./dagger_generator.py --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
#[ $? -ne 0 ] &&  exit 1
#cp graph.json dagger-payloads/linear10-graph.json
#cp dag.png $RESULTS_DIR/linear10-graph.png
#run -- all policy omitted because of deadlock
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS on Linear 10 with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/linear10-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
  [ $? -ne 0 ] && echo "Linear 10 Failed with Policy: $POLICY" &&  exit 1
  #archive result
  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/linear-10-$POLICY-$SYSTEM.csv
  #plot timeline with gantt
  python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph.json --timeline=$RESULTS_DIR/linear-10-$POLICY-$SYSTEM.csv --combined-out=$GRAPHS_DIR/linear-10-$POLICY-$SYSTEM.pdf # --drop="Initialize-0,Initialize-1" #--title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"
  [ $? -ne 0 ] && echo "Failed Combined Plotting of Linear 10 with Policy: $POLICY" &&  exit 1
done

# Parallel 2-by-10
#echo "*******************************************************************"
#echo "*                          Parallel 2by10                         *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
##[ $? -ne 0 ] &&  exit 1
#cp graph.json dagger-payloads/parallel-2by10-graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS on Parallel 2by10 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/parallel-2by10-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="2" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
#  [ $? -ne 0 ] && echo "Parallel 2by10 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/parallel-2by10-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done
#
#echo "*******************************************************************"
#echo "*                          Parallel 5by100                        *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/parallel-5by100-graph.json
##cp dag.png $RESULTS_DIR/parallel-5by100-graph.png
#cp dagger-payloads/parallel-5by100-graph.json graph.json ; cat graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS on Parallel 5by100 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="parallel-5by100-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="ijk" --duplicates="5" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=100 --num-tasks=100 --min-width=1 --max-width=1
#  [ $? -ne 0 ] && echo "Parallel 5by100 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/parallel-5by100-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

echo "*******************************************************************"
echo "*                          Diamond 10                             *"
echo "*******************************************************************"
#diamond 10
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
#[ $? -ne 0 ] &&  exit 1
#cp graph.json dagger-payloads/diamond-10-graph.json
#cp dag.png $RESULTS_DIR/diamond-10-graph.png
cp dagger-payloads/diamond-10-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS on Diamond 10 with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/diamond-10-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=10 --min-width=10 --max-width=10 --sandwich
  [ $? -ne 0 ] && echo "Diamond 10 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/diamond-10-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
  #plot timeline with gantt
  python ./gantt/gantt.py --dag=./dagger-payloads/diamond-10-graph.json --timeline=$RESULTS_DIR/diamond-10-$POLICY-$SYSTEM.csv --combined-out=$GRAPHS_DIR/diamond-10-$POLICY-$SYSTEM.pdf
  [ $? -ne 0 ] && echo "Failed Combined Plotting of Diamond 10 with Policy: $POLICY" &&  exit 1
done

#echo "*******************************************************************"
#echo "*                          Diamond 100                            *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=100 --min-width=100 --max-width=100 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/diamond-100-graph.json
##cp dag.png $RESULTS_DIR/diamond-100-graph.png
#cp dagger-payloads/diamond-100-graph.json graph.json ; cat graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS on Diamond 100 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/diamond-100-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=100 --min-width=100 --max-width=100 --sandwich
#  [ $? -ne 0 ] && echo "Diamond 100 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/diamond-100-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

#echo "*******************************************************************"
#echo "*                          Diamond 1000                           *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/diamond-1000-graph.json
##cp dag.png $RESULTS_DIR/diamond-1000-graph.png
#cp dagger-payloads/diamond-1000-graph.json graph.json ; cat graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS Diamond 1000 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/diamond-1000-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=1 --num-tasks=1000 --min-width=1000 --max-width=1000 --sandwich
#  [ $? -ne 0 ] && echo "Diamond 1000 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/diamond-1000-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

echo "*******************************************************************"
echo "*                          Chainlink 25                           *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
#[ $? -ne 0 ] &&  exit 1
#cp graph.json dagger-payloads/chainlink-25-graph.json
#cp dag.png $RESULTS_DIR/chainlink-25-graph.png
cp dagger-payloads/chainlink-25-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS on Chainlink 25 with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/chainlink-25-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=50 --min-width=1 --max-width=2 --sandwich
  [ $? -ne 0 ] && echo "Chainlink 25 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/chainlink-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
  #plot timeline with gantt
  python ./gantt/gantt.py --dag=./dagger-payloads/chainlink-25-graph.json --timeline=$RESULTS_DIR/chainlink-25-$POLICY-$SYSTEM.csv --combined-out=$GRAPHS_DIR/chainlink-25-$POLICY-$SYSTEM.pdf
  [ $? -ne 0 ] && echo "Failed Combined Plotting of Chainlink 25 with Policy: $POLICY" &&  exit 1
done

#echo "*******************************************************************"
#echo "*                          Galaga 25                              *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/galaga-25-graph.json
##cp dag.png $RESULTS_DIR/galaga-25-graph.png
#cp dagger-payloads/galaga-25-graph.json graph.json ; cat graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS on Galaga 25 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/galaga-25-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich
#  [ $? -ne 0 ] && echo "Galaga 25 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/chainlink-25-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

echo "*******************************************************************"
echo "*                          Tangled 25                             *"
echo "*******************************************************************"
#./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3
#[ $? -ne 0 ] &&  exit 1
#cp graph.json dagger-payloads/tangled-25-graph.json
#cp dag.png $RESULTS_DIR/tangled-25-graph.png
cp dagger-payloads/tangled-25-graph.json graph.json ; cat graph.json
for POLICY in roundrobin depend profile random any all
do
  echo "Running IRIS on Tangled 25 with Policy: $POLICY"
  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/tangled-25-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=25 --min-width=1 --max-width=12 --sandwich
  [ $? -ne 0 ] && echo "Tangled 25 Failed with Policy: $POLICY" &&  exit 1
  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/tangled-25-$POLICY-$SYSTEM.csv
  [ $? -ne 0 ] &&  exit 1
  #plot timeline with gantt
  python ./gantt/gantt.py --dag=./dagger-payloads/tangled-25-graph.json --timeline=$RESULTS_DIR/tangled-25-$POLICY-$SYSTEM.csv --combined-out=$GRAPHS_DIR/tangled-25-$POLICY-$SYSTEM.pdf
  [ $? -ne 0 ] && echo "Failed Combined Plotting of Tangled 25 with Policy: $POLICY" &&  exit 1
done

#echo "*******************************************************************"
#echo "*                           Brain 1000                            *"
#echo "*******************************************************************"
##./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --concurrent-kernels="ijk:3" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=10 --cdf-std-dev=5 --skips=10
##[ $? -ne 0 ] &&  exit 1
##cp graph.json dagger-payloads/brain-1000-graph.json
##cp dag.png $RESULTS_DIR/brain-100-graph.png
#cp dagger-payloads/brain-1000-graph.json graph.json ; cat graph.json
#for POLICY in roundrobin depend profile random any all
#do
#  echo "Running IRIS on Brain 1000 with Policy: $POLICY"
#  IRIS_HISTORY=1 ./dagger_runner --graph="dagger-payloads/brain-1000-graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256 --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=25 --num-tasks=1000 --min-width=1 --max-width=50 --sandwich
#  [ $? -ne 0 ] && echo "Brain 1000 Failed with Policy: $POLICY" &&  exit 1
#  mv dagger_runner-$SYSTEM*.csv $RESULTS_DIR/brain-1000-$POLICY-$SYSTEM.csv
#  [ $? -ne 0 ] &&  exit 1
#done

#TODO: add a mixed kernels test
#save
#rm -rf linear-10-results; mkdir -p linear-10-results; mv linear-10-*.csv linear-10-results
#echo "All results logged into ./linear-10-results"
exit 0
