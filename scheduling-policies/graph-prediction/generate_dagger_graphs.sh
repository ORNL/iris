#!/bin/bash

echo "Running DAGGER evaluation.... (graph figures can be found in $GRAPHS_DIR)"

#Only run DAGGER once to generate the payloads to test the systems (we want to compare the scheduling algorithms over different systems, and so we should fix the payloads over the whole experiment)
#remove the dagger-payloads directory to regenerate payloads
if ! [ -d dagger-payloads ] ; then
  echo "Generating DAGGER payloads (delete this directory to regenerate new DAG payloads)..."
  mkdir -p dagger-payloads
  for SIZE in ${SIZES[@]}
  do
    echo "*******************************************************************"
    echo "*                          Linear $SIZE                           *"
    echo "*******************************************************************"
    ./dagger_generator.py --kernels="ijk" --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=1 $USE_DATA_MEMORY
    [ $? -ne 0 ] &&  exit 1
    cat graph.json
    cp graph.json dagger-payloads/linear$SIZE-graph.json
    cp dag.pdf $GRAPHS_DIR/linear$SIZE-graph.pdf
  done
  for SIZE in ${SIZES[@]}
  do
    #if $SIZE -eq 700 ; then
    #  SIZE=350
    #fi
    echo "*******************************************************************"
    echo "*                          Diamond $SIZE                          *"
    echo "*******************************************************************"
    let width=$SIZE-2
    ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=1 --num-tasks=$width --min-width=$width --max-width=$width --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich  $USE_DATA_MEMORY --concurrent-kernels="ijk:$width"
    [ $? -ne 0 ] &&  exit 1
    cat graph.json
    cp graph.json dagger-payloads/diamond$SIZE-graph.json
    cp dag.pdf $GRAPHS_DIR/diamond$SIZE-graph.pdf
  done
  for SIZE in ${SIZES[@]}
  do
    echo "*******************************************************************"
    echo "*                          Chainlink $SIZE                        *"
    echo "*******************************************************************"
    let depth=$SIZE/2
    ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=$depth --num-tasks=$SIZE --min-width=1 --max-width=2 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --cdf-mean=1.5 --cdf-std-dev=0 --sandwich $USE_DATA_MEMORY --concurrent-kernels="ijk:2" 
    [ $? -ne 0 ] &&  exit 1
    cat graph.json
    cp graph.json dagger-payloads/chainlink$SIZE-graph.json
    cp dag.pdf $GRAPHS_DIR/chainlink$SIZE-graph.pdf
  done
  for SIZE in ${SIZES[@]}
  do
    echo "*******************************************************************"
    echo "*                          Tangled $SIZE                          *"
    echo "*******************************************************************"
    ./dagger_generator.py --kernels="ijk" --kernel-split='100' --depth=$SIZE --num-tasks=$SIZE --min-width=1 --max-width=12 --buffers-per-kernel="ijk:rw r r" --kernel-dimensions="ijk:2" --sandwich --cdf-mean=2 --cdf-std-dev=0 --skips=3 $USE_DATA_MEMORY --concurrent-kernels="ijk:12"
    [ $? -ne 0 ] &&  exit 1
    cat graph.json
    cp graph.json dagger-payloads/tangled$SIZE-graph.json
    cp dag.pdf $GRAPHS_DIR/tangled$SIZE-graph.pdf
  done
fi
