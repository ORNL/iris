"""
Tool to generate the combined super heatmap of % occupancy of timeline and schedules
"""

class Heatmap():

    def __init__(self, device_colour_palette=None, use_device_background_colour=False):
        self.device_colour_palette = device_colour_palette
        self.use_device_background_colour = use_device_background_colour

    def plotHeatmap(self,directory, output, width, height, custom_policy_rename, subtract_gnn_overhead, **kargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
        import pandas as pandas
        from bokeh import palettes
        from load_data import LoadData
        dataloader = LoadData(directory,custom_policy_rename, subtract_gnn_overhead)
        data = dataloader.GetData()
        data = pandas.DataFrame.from_records(data)

        # statistical reduction (median and variance):
        data = data.groupby(['system-and-policy','dag']).agg(["median","var"]).reset_index()
        data = data.fillna(0)
        #statistical reduction (median)
        #data = data.groupby(['system-and-policy','dag']).median().reset_index()
        data = data.sort_values('system-and-policy')
        heatmap_data = data.pivot(index='system-and-policy', columns='dag')
        #reorder columns to be sensible (linear10, linear25, linear100) etc
        from natsort import humansorted
        heatmap_data = heatmap_data[humansorted(heatmap_data.columns)]

        print(heatmap_data)

        # generate heatmap
        fig, ax = plt.subplots(figsize=(width,height))
        cmap = sns.cm.rocket_r
        heatmap_plot = sns.heatmap(heatmap_data['duration']['median'],annot=round(heatmap_data['duration']['median'],3).astype(str)+'±'+round(heatmap_data['duration']['var'],3).astype(str),fmt="",cmap=cmap)
        heatmap_plot.collections[0].colorbar.set_label("median time to completion (sec)")
        plt.xlabel('DAG', fontsize = 15)
        if dataloader.OnlyOneSystem():
            plt.ylabel('Policy', fontsize = 15)
        else:
            plt.ylabel('System and Policy', fontsize = 15)
        fig = heatmap_plot.get_figure()
        fig.tight_layout()
        # save heatmap to disk
        fig.savefig(output)


if __name__ == '__main__':
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog='IRIS Heatmap Plotter',
        description='This program takes a directory of dagger runtime results and collates them into a single heatmap.')
    parser.add_argument('--directory',dest="directory",type=str,help="directory path to results (will result in a directory listing of .csv)")
    parser.add_argument('--output-file',dest="outputfile",type=str,help="filepath for where you would like to store the heatmap plot (.pdf/.png)")
    parser.add_argument("--width",dest="width",type=int, default=20, required=False)
    parser.add_argument("--height",dest="height",type=int, default=15, required=False)
    parser.add_argument("--custom-rename",dest="customname", type=str, default=None, required=False)
    parser.add_argument("--subtract-gnn-overhead",dest="subtractgnnoverhead", default=False, required=False, action='store_true')

    args = parser.parse_args()
    directory = args.directory
    output_file = args.outputfile

    if directory is None:
        directory = "../dagger-results"
        print("No directory provided... using {}".format(directory))
    if output_file is None:
        print("Incorrect Arguments. Please provide *at least* one output filepath (--output-file)")
        sys.exit(1)

    Heatmap.plotHeatmap(None,directory,output_file,args.width,args.height,args.customname,args.subtractgnnoverhead)

