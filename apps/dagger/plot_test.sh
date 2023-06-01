#!/bin/bash

#dag
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph.json --timeline=./dagger-figures/linear-10-random-explorer.csv --dag-out=just-dag.pdf --title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"

#timeline
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph.json --timeline=./dagger-figures/linear-10-random-explorer.csv --timeline-out=just-timeline.pdf --title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"

#combined
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph.json --timeline=./dagger-figures/linear-10-random-explorer.csv --combined-out=just-combined.pdf --drop="Initialize-0,Initialize-1" #--title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"

#all
python ./gantt/gantt.py --dag=./dagger-payloads/linear10-graph.json --timeline=./dagger-figures/linear-10-random-explorer.csv --combined-out=all-combined.pdf --timeline-out=all-timeline.pdf --dag-out=all-dag.pdf --title-string="Linear 10 dataset with RANDOM scheduling policy" --drop="Init"

tar -cvf plot_test_graphs.tar just-dag.pdf just-timeline.pdf just-combined.pdf all-dag.pdf all-timeline.pdf all-combined.pdf
rm just-dag.pdf just-timeline.pdf just-combined.pdf all-dag.pdf all-timeline.pdf all-combined.pdf
echo "results stored in the plot_test_graphs.tar tarball"
