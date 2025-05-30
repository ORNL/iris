#!/usr/bin/env python

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
df = pd.read_csv("dagger-results/lws_times.csv")

print("Local workgroup execution times in csv file:", df)
sns.barplot(data=df,x='size',y='secs',hue='dim')
plt.xticks(rotation=45)
plt.show()
plt.savefig('dagger-graphs/lws_times.pdf')
