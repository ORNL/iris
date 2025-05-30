#!/bin/bash

# DAGGER just symbolically linked here for ease of use.
#ln -s ../../apps/dagger dagger

# Run the benchmark evaluation
./run-baseline-evaluation.sh

# Plot the result
python3 ./dagger/gantt/heatmap.py --output-file baseline-heatmap.pdf --directory ./results/ --height=5

