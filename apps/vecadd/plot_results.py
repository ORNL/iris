import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from statistics import median

assert 3 <= len(sys.argv) <= 4
dir_str = sys.argv[1]
title_str = sys.argv[2]
profiled = False
if len(sys.argv) == 4:
  profiled = bool(sys.argv[3])
#todo add stacked plot
targetdir="results/"+dir_str+"/"
#list the files
filelist = os.listdir(targetdir)
#read them into pandas
df = pd.DataFrame()
for file in filelist:
  df[file.rstrip('.csv')] = pd.read_table(os.path.join(targetdir,file),header=None)[0]#no probs (only one column per file
#reorganize columns numerically
df.columns = df.columns.astype(int)
df = df.sort_index(axis=1)

if profiled:
  old_df = df.copy()
  #do a reduction on the data for the same old box-and-whisker plot
  df = pd.DataFrame()
  for i in old_df:
    vec = []
    for j in range(1,len(old_df[i])):
      vec.append(int(str.split(old_df[i][j],',')[0])) # get the walltimes
    df[i] = vec
  #also collect the data for the stacked boxplot
  x = []
  submit = np.array([])
  kernel = np.array([])
  unaccounted = np.array([])
  for i in old_df:
    x.append(str(i))
    ts = []
    tk = []
    tu = []
    for j in range(1,len(old_df[i])):
      ts.append(int(str.split(old_df[i][j],',')[1])) # get the full list of submit times
      tk.append(int(str.split(old_df[i][j],',')[2])) # get the full list of kernel times
      tu.append(int(str.split(old_df[i][j],',')[3])) # get the full list of unaccounted times
    #record the median time (in microseconds)--- some down-sampling must occur
    submit = np.append(submit,median(ts)*0.001)
    kernel = np.append(kernel,median(tk)*0.001)
    unaccounted = np.append(unaccounted,median(tu)*0.001)
  # plot bars in stack manner
  plt.figure(figsize=(11.7,8.3))
  plt.bar(x, submit, color='r')
  plt.bar(x, kernel, bottom=submit, color='g')
  plt.bar(x, unaccounted, bottom=submit+kernel, color='y')
  plt.title(title_str)
  plt.legend(["Submit", "Kernel", "Unaccounted"])
  plt.xlabel('Payload size (# elements)')
  plt.xticks(rotation = 45)
  plt.ylabel('Time ('r'$\mu$s)')
  plt.ylim([0, 600])
  plt.savefig("figures/"+dir_str+"_bar.pdf")

#convert times from nanoseconds to microseconds
df = df*0.001

#plot box and whisker plot (box per column)
plt.figure(figsize=(17,1))
df.plot(kind='box')
plt.title(title_str)
plt.ylabel('Time ('r'$\mu$s)')
plt.xlabel('Vector size (# elements)')
plt.xticks(rotation = 45)
plt.ylim([0, 1200])
plt.savefig("figures/"+dir_str+".pdf")
