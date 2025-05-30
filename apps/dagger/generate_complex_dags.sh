#!/bin/bash
source ./setup.sh

if [ "$SYSTEM" = "leconte" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "equinox" ] ; then
  rm -f *.csv ; make dagger_runner kernel.ptx
elif [ "$SYSTEM" = "explorer" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip
elif [ "$SYSTEM" = "radeon" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip;
elif [ "$SYSTEM" = "zenith" ] ; then
  rm -f *.csv ; make dagger_runner kernel.hip kernel.ptx kernel.openmp.so;
else
  echo "Unknown system." && exit
fi

#exit if the last program run wasn't successful
[ $? -ne 0 ] && exit

#don't proceed if the target failed to build
if ! [ -f dagger_runner ] ; then
  exit
fi

#ensure libiris.so is in the shared library path
  echo "ADDING $HOME/.local/lib64 to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH
export IRIS_ARCHS=cuda,openmp
echo "*******************************************************************"
echo "*             Shared Linear-10 (IRIS Data-Memory)                 *"
echo "*******************************************************************"
##build linear-50 DAG
./dagger_generator.py --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --num-memory-objects=5 --use-data-memory
[ $? -ne 0 ] && exit
cat graph.json
for POLICY in random roundrobin sdq ftf
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 IRIS_HISTORY_FILE=timeline_log.csv ./dagger_runner --graph="graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --num-memory-objects=5 --use-data-memory
    [ $? -ne 0 ] && exit
done
python ./gantt/gantt.py --dag=./graph.json --timeline=./timeline_log.csv --combined-out=shared-linear-dmem.pdf --title-string="Shared IRIS Data-Memory on Linear-10 dataset with any scheduling policy" #--drop="Init"
[ $? -ne 0 ] && exit
echo "*******************************************************************"
echo "*                      Shared Linear-10                           *"
echo "*******************************************************************"
##build linear-50 DAG
./dagger_generator.py --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --num-memory-objects=5
[ $? -ne 0 ] && exit
cat graph.json
for POLICY in random roundrobin sdq ftf
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 IRIS_HISTORY_FILE=timeline_log.csv ./dagger_runner --graph="graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --num-memory-objects=5
    [ $? -ne 0 ] && exit
done
python ./gantt/gantt.py --dag=./graph.json --timeline=./timeline_log.csv --combined-out=shared-linear.pdf --title-string="Shared Memory on Linear-10 dataset with roundrobin scheduling policy" #--drop="Init"
[ $? -ne 0 ] && exit
echo "*******************************************************************"
echo "*                    Linear-10 (IRIS Data-Memory)                 *"
echo "*******************************************************************"
##build linear-50 DAG
./dagger_generator.py --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory
[ $? -ne 0 ] && exit
cat graph.json
for POLICY in random roundrobin sdq ftf
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 IRIS_HISTORY_FILE=timeline_log.csv ./dagger_runner --graph="graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 --use-data-memory
    [ $? -ne 0 ] && exit
done
python ./gantt/gantt.py --dag=./graph.json --timeline=./timeline_log.csv --combined-out=linear-dmem.pdf --title-string="IRIS Data-Memory on Linear-10 dataset with roundrobin scheduling policy" #--drop="Init"
[ $? -ne 0 ] && exit
echo "*******************************************************************"
echo "*                             Linear-10                           *"
echo "*******************************************************************"
##build linear-50 DAG
./dagger_generator.py --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1
[ $? -ne 0 ] && exit
cat graph.json
for POLICY in roundrobin
do
  echo "Running IRIS with Policy: $POLICY"
  IRIS_HISTORY=1 IRIS_HISTORY_FILE=timeline_log.csv ./dagger_runner --graph="graph.json" --logfile="time.csv" --repeats=1 --scheduling-policy="$POLICY" --size=256  --kernels="bigk,process,ijk" --duplicates="0" --buffers-per-kernel="bigk:rw r r,process:rw,ijk:w r r" --kernel-dimensions="bigk:2,process:1,ijk:2" --kernel-split='60,20,20' --depth=10 --num-tasks=10 --min-width=1 --max-width=1 
    [ $? -ne 0 ] && exit
done
python ./gantt/gantt.py --dag=./graph.json --timeline=./timeline_log.csv --combined-out=linear.pdf --title-string="Linear-10 dataset with roundrobin scheduling policy" #--drop="Init"
[ $? -ne 0 ] && exit

