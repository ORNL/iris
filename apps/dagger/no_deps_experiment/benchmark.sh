#!/bin/bash

cd ..
source ./setup.sh
make dagger_runner kernel.ptx
cd no_deps_experiment
cp ../kernel.ptx .

#just the included paper DAG
../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=10 --num-tasks=64 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashloaddemo.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=32 --no-deps --no-flush
export IRIS_ARCHS=cuda
#asynchronous
IRIS_PROFILE=1 IRIS_ASYNC=1 IRIS_MALLOC_ASYNC=0 IRIS_NSTREAMS=9 IRIS_NCOPY_STREAMS=3 IRIS_HISTORY=1 IRIS_HISTORY_FILE=mashloaddemo.csv ../dagger_runner --graph="mashloaddemo.json"  --repeats=1 --scheduling-policy="roundrobin" --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2"  --size=2048 --concurrent-kernels="ijk:14" --use-data-memory
#python ../gantt/gantt.py --dag=mashloaddemo.json --timeline=mashloaddemo.csv --dag-out=64dag.pdf --no-show-kernel-legend --no-show-task-legend --no-show-node-labels
dot -T pdf dagger_runner-milan2.ftpn.ornl.gov-*.dot -o dag.pdf

exit

#now for the actual experiment --- generate the DAGGER payloads
GENERATE_PAYLOADS=1
if [ $GENERATE_PAYLOADS -eq 1 ]
then
  #64
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=10 --num-tasks=64 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-64-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=32

  #128
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=20 --num-tasks=128 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-128-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=64

  #256
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=40 --num-tasks=256 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-256-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=128

  #512
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=80 --num-tasks=512 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-512-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=256

  #1024
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=160 --num-tasks=1024 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-1024-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=512

  #2048
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=320 --num-tasks=2048 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-2048-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=1024

  #4096
  ../dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=640 --num-tasks=4096 --min-width=7 --max-width=7 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --use-data-memory --concurrent-kernels="ijk:14" --skips=3 --cdf-mean=2 --cdf-std-dev=0 --graph="mashload-4096-graph.json" --use-data-memory --handover-in-memory-shuffle --num-memory-shuffles=2048

fi

#then run each of the payloads
export SIZES=(64 128 256 512 1024 2048 4096);

export IRIS_ARCHS=cuda
for SIZE in ${SIZES[@]}
do
  #asynchronous
  IRIS_ASYNC=1 IRIS_MALLOC_ASYNC=0 IRIS_NSTREAMS=9 IRIS_NCOPY_STREAMS=3 IRIS_HISTORY=1 IRIS_HISTORY_FILE=mashload-$SIZE-async-timeline.csv ../dagger_runner --graph="mashload-$SIZE-graph.json" --repeats=1 --scheduling-policy="roundrobin" --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2"  --size=2048 --concurrent-kernels="ijk:14" --use-data-memory
  python ../gantt/gantt.py --dag=mashload-$SIZE-graph.json --timeline=mashload-$SIZE-async-timeline.csv --combined-out=mashload-$SIZE-async-out.pdf
  #synchronous
  IRIS_ASYNC=0 IRIS_MALLOC_ASYNC=0 IRIS_NSTREAMS=9 IRIS_NCOPY_STREAMS=3 IRIS_HISTORY=1 IRIS_HISTORY_FILE=mashload-$SIZE-sync-timeline.csv ../dagger_runner --graph="mashload-$SIZE-graph.json" --repeats=1 --scheduling-policy="roundrobin" --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2"  --size=2048 --concurrent-kernels="ijk:14" --use-data-memory
  python ../gantt/gantt.py --dag=mashload-$SIZE-graph.json --timeline=mashload-$SIZE-sync-timeline.csv --combined-out=mashload-$SIZE-sync-out.pdf
done

mkdir -p results
mv mashload-* results

exit
