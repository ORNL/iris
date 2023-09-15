"""
Tool to generate the combined super heatmap of % occupancy of timeline and schedules
"""

class Heatmap():

    def __init__(self, device_colour_palette=None, use_device_background_colour=False):
        self.device_colour_palette = device_colour_palette
        self.use_device_background_colour = use_device_background_colour

    def plotHeatmap(self,directory, output, **kargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
        import pandas as pandas
        from bokeh import palettes

        # load in all results
        print("Loading results from: {}".format(directory))
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        #TODO: delete
        ##batch renaming (delete a leading '-' character from all files)
        #import shutil
        #for f in files:
        #    shutil.move(os.path.join(directory,f),os.path.join(directory, f.replace('-','',1)))
        data = []
        for f in files:
            if "datamemlinear10" in f: continue #skip the deep-dive result
            content = pandas.read_csv(os.path.join(directory, f))
            dag,schedule,system,*_ = f.replace('.csv','').split('-')
            start = min(content['start'])
            end = max(content['end'])
            data.append({"system-and-policy":"{}-{}".format(system,schedule), "duration":end-start, "dag":dag})

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

        # TODO: ensure rows by cols are sensible---may have to group (by colour) the system, but certainly need to list the dagger-payload
        # generate heatmap
        fig, ax = plt.subplots(figsize=(20,15))
        cmap = sns.cm.rocket_r
        heatmap_plot = sns.heatmap(heatmap_data['duration']['median'],annot=round(heatmap_data['duration']['median'],3).astype(str)+'±'+round(heatmap_data['duration']['var'],3).astype(str),fmt="",cmap=cmap)
        heatmap_plot.collections[0].colorbar.set_label("median time to completion (sec)")
        plt.xlabel('DAG', fontsize = 15)
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

    args = parser.parse_args()
    directory = args.directory
    output_file = args.outputfile
    if directory is None:
        directory = "../dagger-results"
        print("No directory provided... using {}".format(directory))
    if output_file is None:
        print("Incorrect Arguments. Please provide *at least* one output filepath (--output-file)")
        sys.exit(1)

    Heatmap.plotHeatmap(None,directory,output_file)

