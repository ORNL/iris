import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

assert len(sys.argv) == 3
dir_str = sys.argv[1]
title_str = sys.argv[2]

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

#convert times from nanoseconds to microseconds
df = df*0.001

#plot boxplot (box per column)
plt.figure(figsize=(17,1))
df.plot(kind='box')
plt.title(title_str)
plt.ylabel('Time ('r'$\mu$s)')
plt.xlabel('Vector size (# elements)')
plt.xticks(rotation = 45)
#plt.ylim([0, 1200])
plt.savefig("figures/"+dir_str+".pdf")
