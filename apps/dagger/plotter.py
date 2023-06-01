#!/usr/bin/env python

import os
import sys
import importlib
import pandas as pandas
import gantt.gantt as gantt
importlib.reload(gantt)
g = gantt.Gantt()

if len(sys.argv) < 3:
  print("Incorrect arguments. Please provide ./plotter.py <log.csv> <output.png> [optional:title string] [optional:elements to drop, eg:(\"Init,H2D\")]\n Aborting...")
  sys.exit(0)

# get the minimum and maximum time values---to show a consistent time range in the plot
mint = sys.float_info.max
maxt = sys.float_info.min
source_file = sys.argv[1]
output_file = sys.argv[2]
print("length of arguments"+str(len(sys.argv)))
title_string=""
if len(sys.argv) >= 4:
  title_string = sys.argv[3]
  print("using title_string "+title_string)
dropsy = []
if len(sys.argv) >= 5:
  dropsy = str(sys.argv[4]).split(',')
  print("and dropping")
  print(dropsy)

x = pandas.read_csv(source_file)
# drop entries without
x = x.dropna()
print(list(set(x['acclname'])))
# drop Init
# x = x[x.taskname != 'Init']
mint = min(mint, min(x['start']))
maxt = max(maxt, max(x['end']))

window_buffer = (maxt-mint)/10
time_range = [mint-window_buffer, maxt+window_buffer]

# generate the plot
g.saveGanttChart(source_file,file=output_file,drop=dropsy,title=title_string,time_range=time_range,outline=False)

