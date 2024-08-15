"""
Tool to generate a performance comparison lineplot between schedules
"""

class LinePlotComparison():
    def __init__(self):
        print("initialized LinePlotComparison")

    def plotLinePlot(self, directory, output, width, height, target_schedule, custom_policy_rename, subtract_gnn_overhead,  statistic="median", unit="ms"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
        import pandas as pandas
        from bokeh import palettes
        from load_data import LoadData
        dataloader = LoadData(directory,custom_policy_rename, subtract_gnn_overhead)
        data = dataloader.GetData()
 
        if target_schedule is None:
            #TODO: if it's none, just use the first policy as the target to compare against
            print("TODO")

        data = pandas.DataFrame.from_records(data)
        data = data.sort_values(by="dag")

        target_data = data[data['system-and-policy']==target_schedule]
        data = data[data['system-and-policy']!=target_schedule] 
        for dag in set(data['dag']):
            x = data[data['dag']==dag]
            y = x.groupby(["system-and-policy","dag"]).agg({"duration":statistic})
            entry_with_minimum_time = x['duration'].idxmin(0)
            best_policy = x.loc[entry_with_minimum_time]
            target_data = target_data._append(x[x["system-and-policy"] == best_policy['system-and-policy']])
        data = target_data.sort_values(by="dag")
        from natsort import index_humansorted
        import numpy as np
        data = data.sort_values(by="dag", key=lambda x: np.argsort(index_humansorted(data["dag"])))
        fig, ax = plt.subplots(figsize=(width,height))
        boxplot = sns.boxplot(data=data, x="dag", y="duration", hue="system-and-policy")
        plt.ylabel(statistic+" time to completion ("+unit+")", fontsize = 15)
        plt.xlabel("DAG", fontsize = 15)
        plt.legend(title='Policy')
        fig = boxplot.get_figure()
        fig.savefig(output) 

if __name__ == '__main__':
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog='IRIS Line Plot Comparison',
        description='This program takes a directory of dagger runtime results and collates them into a single lineplot.')
    parser.add_argument('--directory',dest="directory",type=str,help="directory path to results (will result in a directory listing of .csv)")
    parser.add_argument('--output-file',dest="outputfile",type=str,help="filepath for where you would like to store the heatmap plot (.pdf/.png)")
    parser.add_argument("--width",dest="width",type=int, default=20, required=False)
    parser.add_argument("--height",dest="height",type=int, default=15, required=False)
    parser.add_argument("--target",dest="target",type=str, default=None, help="the target policy to compare against")
    parser.add_argument("--custom-rename",dest="customname", type=str, default=None, required=False)
    parser.add_argument("--subtract-gnn-overhead",dest="subtractgnnoverhead", default=False, required=False, action='store_true')
    parser.add_argument("--statistic",dest="stat", type=str, default="median", required=False)
    parser.add_argument("--units",dest="units", type=str, default="ms", required=False)

    args = parser.parse_args()
    directory = args.directory
    output_file = args.outputfile

    if directory is None:
        directory = "../dagger-results"
        print("No directory provided... using {}".format(directory))
    if output_file is None:
        print("Incorrect Arguments. Please provide *at least* one output filepath (--output-file)")
        sys.exit(1)

    LinePlotComparison.plotLinePlot(None,directory,output_file,args.width,args.height,args.target,args.customname,args.subtractgnnoverhead,args.stat,args.units)
    print("done.")

