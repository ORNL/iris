"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)
And implementation of plotting the same schedules on the original DAG---the colour of the devices match the gantt/timeline
"""

class Gantt():

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pandas
    from bokeh import palettes

    def __init__(self, device_colour_palette=None, use_device_background_colour=False):
        self.device_colour_palette = device_colour_palette
        self.use_device_background_colour = use_device_background_colour

    def _createGanttChart(self, timings, title=None, edgepalette=None, insidepalette=None, time_range=None, zoom=False, drop=[], outline=True, inner_label=False, timeline_output_file=None):
        import numpy as np
        timings['taskname'] = np.where(~timings['taskname'].isnull(),timings['taskname'],timings['type'])

        for d in drop:
          timings = timings[timings.taskname != d]

        from natsort import humansorted
        processors = sorted(list(set(timings['acclname'])),reverse=True)

        proc_names = [p for p in processors]
        
        # color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
        # Assign colors based on the palette or default to a palette.
        unique_kernels = humansorted(set(timings['taskname']))
        if insidepalette is None:
            unique_kernels_count = len(unique_kernels)
            if unique_kernels_count < 9:
                insidepalette = self.palettes.brewer['Pastel1'][9]
            else:
                insidepalette = []
                for i in range(unique_kernels_count):
                    #use a unique and consistent colour for all internal D2H and H2D memory transfers
                    if unique_kernels[i] == "D2H":
                        insidepalette.append(self.palettes.brewer['Greys'][9][4])
                    elif unique_kernels[i] == "H2DNP":
                        insidepalette.append(self.palettes.brewer['Greys'][9][0])
                    elif unique_kernels[i] == "H2D":
                        insidepalette.append(self.palettes.brewer['Greys'][9][0])
                    else:
                        insidepalette.append(self.palettes.all_palettes['Turbo'][256][int(i*(256.0/unique_kernels_count))])
        elif insidepalette == 'single':
            insidepalette = ['white' for i in unique_kernels]

        insidephash = {}
        for i,k in enumerate(unique_kernels):
            insidephash[k] = insidepalette[i]

        if edgepalette is None:
            if outline:
              edgepalette = ['black' for i in unique_kernels]
            else:
              edgepalette = insidepalette

        edgephash = {}
        for i,k in enumerate(unique_kernels):
            edgephash[k] = edgepalette[i]


        ilen=len(processors)
        pos = self.np.arange(0.5,ilen*0.5+0.5,0.5)
        fig = self.plt.figure(figsize=(12,6)) # orig
        ax = fig.add_subplot(111)
        used_labels = []
        for idx, proc in enumerate(processors):
            for idy in timings[timings['acclname'] == proc].index.tolist():
                job = timings.loc[[idy]]

                #extra logic to avoid duplicate labels
                fresh_label = job['taskname'].values[0] not in used_labels
                used_labels.append(job['taskname'].values[0])

                ax.barh((idx*0.5)+0.5, job['end'] - job['start'], left=job['start'], height=0.3, align='center', edgecolor=edgephash[job['taskname'].values[0]], color=insidephash[job['taskname'].values[0]], alpha=0.95,label=job['taskname'].values[0] if fresh_label else "__no_legend__")
                if inner_label:
                    #uncomment the following to have the kernel name directly onto the bars rather than as a label
                    ax.text(job['start'] + (.05*(job['end']-job['start'])), (idx*0.5)+0.5 - 0.03125, job['taskname'].values[0], fontweight='bold', fontsize=9, alpha=0.75, rotation=90)

        locsy, labelsy = self.plt.yticks(pos, proc_names)
        self.plt.ylabel('Devices', fontsize=10)
        self.plt.xlabel('Time (s)', fontsize=10)
        self.plt.setp(labelsy, fontsize = 8)
        #ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
        ax.set_xlim(xmin = -0.1, xmax = max(timings['end'])+0.1)
        #TODO: move this to kwargs
        if self.use_device_background_colour:
            for tl in ax.get_yticklabels():
                txt = tl.get_text()
                tl.set_backgroundcolor(self.device_colour_palette[txt])
                tl.set_text(txt)
        # sort both labels and handles by labels
        handles, labels = ax.get_legend_handles_labels()
        neworder = dict(humansorted(zip(labels, handles)))
        ax.legend(neworder.values(), neworder.keys(), title="Tasks",fontsize=8)#bbox_to_anchor=(0, .5), ncol = 1,

        if time_range and not zoom:
            ax.set_xlim(time_range)
        elif zoom:
            padding = (max(timings['start'])-min(timings['end']))/10 #one-tenth of the total span is at the start and end of the zoomed plot
            ax.set_xlim(xmin=min(timings['start'])-padding,xmax=max(timings['end'])+padding)
        ax.grid(color = 'g', linestyle = ':', alpha=0.5)

        ax.invert_yaxis()
        if title:
            self.plt.title(str(title))
        self.plt.tight_layout()

        font = self.font_manager.FontProperties(size='small')
        return fig

    def plotGanttChart(self,timing_log,title=None, **kargs):
        """
            Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
        """  
        timing_content = None
        if type(timing_log) is str:
            timing_content = self.pandas.read_csv(timing_log)
        elif type(timing_log) is self.pandas.core.frame.DataFrame:
            timing_content = timing_log
        else:
            print("File format for {} is {} and unsupported, please provide the url to the timing log instead.".format(timing_log,type(timing_log)))
            return
        fig = self._createGanttChart(timing_content,title, **kargs)
        self.plt.show()
        if timeline_output_file is not None:
          self.plt.savefig(timeline_output_file)
          print("timeline written to "+str(timeline_output_file))
        return fig

class DAG():

    def __init__(self, dag_file, timeline_file):
        self.tasks, self.edges = self.getJsonToTask(dag_file)
        self.timeline = self.getTimelineFromFile(timeline_file)
        self.device_colour_palette = None

    def getJsonToTask(self, dag_file):
        tasks, edges = [],[]
        import json
        f = open(dag_file)
        data = json.load(f)
        for task in data['iris-graph']['graph']['tasks']:
            if "transfer" not in task['name']:
              try:
                kernel_name = task['commands'][0]['kernel']['name']
              except:
                kernel_name = "memory_transfer"
              tasks.append({'name':task['name'],'kernel':kernel_name})
              for dep in task['depends']:
                  if "transfer" not in dep:
                      edges.append((task['name'],dep))
        return tasks,edges

    def getTimelineFromFile(self,timeline_file):
        from pandas import read_csv
        assert type(timeline_file) is str
        timing_content = read_csv(timeline_file)
        return timing_content

    def plotDag(self, dag_path_plot=None, show_device_legend=True):
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        #from bokeh.palettes import Turbo256
        from bokeh.palettes import Set3
        from math import floor
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib import patches as patch
        #debug to try out a few unique shapes (TODO:delete)
        #kernels = ['ijk', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        #for i,k in enumerate(self.tasks):
        #    self.tasks[i]['kernel'] = kernels[i]
        #add labels to edges
        task_dag = self.tasks
        edges = self.edges
        edge_d = []
        for i, e in enumerate(edges):
            n = ""
            for t in task_dag:
                if t['name'] == e[0]:
                    n = t['name']
                    break
            assert(n != "")
            edge_d.append((e[0],e[1]))#,{"depends":n}))

        dag = nx.DiGraph()
        #generate the set of kernel shapes
        kernels = []
        for t in self.tasks:
          kernels.append(t['kernel'])

        unique_kernels = set(kernels)
        shapes = []
        avail_shapes = {0:'s',1:'o',2:'*',3:'^',4:'v',5:'<',6:'>',7:'h',8:'H',9:'D',10:'d',11:'P',12:"X",13:'1',14:'p',15:'.'}
        assert len(unique_kernels) <= len(avail_shapes.items())
        for d in range(0,len(unique_kernels)):
            shapes.append(avail_shapes[d])
        #associate each kernel name with a shape
        kernel_shapes = {}
        for i,d in enumerate(unique_kernels):
            kernel_shapes[d] = shapes[i]

        #generate the colour palette --- using as many unique colours as there are devices
        unique_devices = set(self.timeline['acclname'])
        #assert len(unique_devices) < 256 #if we have more than 256 we can't use Turbo256
        assert len(unique_devices) < 12 
        palette = []
        for d in range(0,len(unique_devices)):
            #palette.append(Turbo256[d*floor(256 / len(unique_devices))])
            palette.append(Set3[12][d*floor(12 / len(unique_devices))])
        #associate each device name with a colour
        device_colour = {}
        for i,d in enumerate(unique_devices):
            device_colour[d] = palette[i]
        self.device_colour_palette = device_colour #save the colour palette for later--incase the timeline plot should use it
        #if there are edges, create the DAG from it, otherwise just the nodes (there are no edges in the DAG, and so the draw call will fail)
        node_d = [(str(e['name']),{"label":e['name'], "position":(i,0), "marker":kernel_shapes[e['kernel']]}) for i, e in enumerate(task_dag)]
        dag.add_nodes_from(node_d)
        node_labels = {str(n['name']):'{}'.format(n['name']) for i,n in enumerate(task_dag)}
        if edge_d != []:
            dag.add_edges_from(edge_d)
            edge_labels = {(e1,e2):'{}'.format(d) for e1,e2,d in dag.edges(data=True)}
        #plot it
        pos = graphviz_layout(dag,prog='dot')
        #add colour
        node_colours = []
        node_shapes = []
        for n in dag:
            name = None
            for z in task_dag:
              if z['name'] == n:
                name = z['name']
                break
            node_colours.append(device_colour[tuple(self.timeline[self.timeline['taskname'] == name]['acclname'])[0]])
            kernel_name = [ item['kernel'] for item in self.tasks if item['name'] == name ]
            if isinstance(kernel_name,list):
                kernel_name = kernel_name[0]
            node_shapes.append(kernel_shapes[kernel_name])

        fig = plt.figure(figsize=(3,6))
        ax = fig.add_subplot(111)
        nx.draw(dag,pos=pos,labels=node_labels,font_size=8,node_color=node_colours,  ax=ax,node_shape=node_shapes)
        #failed attempt at using generic networkx package---rather than my own modification in python -m pip install "networkx @ git+https://github.com/BeauJoh/networkx.git@main"
        #for i, (n, s) in enumerate(zip(dag, node_shapes)):
        #    nx.draw(dag,pos=pos,labels=node_labels,ax=ax,nodelist=[n],font_size=8,node_color=node_colours[i], node_shape=s)
        #add kernel shape legend
        legend_handles = []
        for i,d in enumerate(unique_kernels):
            legend_handles.append(ax.scatter([], [],color='white', edgecolor='black', marker=kernel_shapes[d], label=d))
        kernel_legend = ax.legend(handles=legend_handles,loc=1,title="Kernels",fontsize=8)
        #add device colour legend
        legend_handles = []
        for i,d in enumerate(unique_devices):
            legend_handles.append(patch.Patch(color=device_colour[d], label=d))
        if show_device_legend:
          #ax.legend(handles=legend_handles,loc=3,title="Devices",fontsize=8)
          ax.legend(handles=legend_handles,loc='upper center', bbox_to_anchor=(0.5, 0.0),title="Devices",fontsize=8)
        plt.gca().add_artist(kernel_legend)
        if dag_path_plot is not None:
            plt.savefig(dag_path_plot)
            print("dag written to "+str(dag_path_plot))
        return fig


class CombinePlots():
    def __init__(self, timeline_file=None,dag_file=None,combined_output_file=None,timeline_output_file=None,dag_output_file=None,title_string=None,drop=None,**kargs):
        assert timeline_file is not None, f"timeline file not provided"
        assert dag_file is not None, f"dag file not provided"
        if combined_output_file is None and timeline_output_file is None and dag_output_file is None:
            print("Error: *at least* one output file must be specified (either combined_output_file, timeline_output_file or dag_output_file.")
            import sys
            sys.exit(1)
        self.timeline_file = timeline_file
        self.dag_file = dag_file
        self.output_file = combined_output_file
        self.timeline_output_file = timeline_output_file
        self.dag_output_file = dag_output_file
        self.drop = drop
        self.title_string = title_string
        self.kargs = kargs
        self.PlotBoth()

    def write_pdf(self,figures):
        from matplotlib.backends.backend_pdf import PdfPages
        fname = self.output_file
        doc = PdfPages(fname)
        for fig in figures:
            fig.savefig(doc, format='pdf')
            print("combined plots written to "+str(fname))
        doc.close()

    def PlotBoth(self):
        x = pandas.read_csv(self.timeline_file)
        # drop entries without
        x = x.dropna()
        #print(list(set(x['acclname'])))
        # drop Init
        # x = x[x.taskname != 'Init']
        # get the minimum and maximum time values---to show a consistent time range in the plot
        mint = sys.float_info.max
        maxt = sys.float_info.min
        mint = min(mint, min(x['start']))
        maxt = max(maxt, max(x['end']))

        window_buffer = (maxt-mint)/10
        time_range = [mint-window_buffer, maxt+window_buffer]

        # generate the dag/graph plot
        dag = DAG(self.dag_file,timeline_file=self.timeline_file)
        right = dag.plotDag(self.dag_output_file,show_device_legend=True)
        # generate the timeline/gantt plot
        use_device_background_colour = False
        if 'use_device_background_colour' in self.kargs:
            use_device_background_colour = self.kargs['use_device_background_colour']
        gantt = Gantt(device_colour_palette=dag.device_colour_palette,use_device_background_colour=use_device_background_colour)
        left = gantt.plotGanttChart(timing_log=self.timeline_file,drop=self.drop,title=self.title_string,time_range=time_range,outline=False,timeline_output_file=self.timeline_output_file)
        if self.output_file is not None:
            self.write_pdf([left, right])
        return

if __name__ == '__main__':
    import os
    import sys
    import pandas as pandas
    import argparse

    parser = argparse.ArgumentParser(
        prog='IRIS Plotter',
        description='This program takes IRIS DAGs (in JSON format) and/or execution timelines (csv) to visually present the Graph of dependencies and timeline GANTT. It shows the impact of IRIS secheduling on different DAG payloads.')
    parser.add_argument('--timeline',dest="timeline",type=str,help="filepath to execution log from IRIS run (.csv)")
    parser.add_argument('--dag',dest="dag",type=str,help="filepath to dag dependency file (.json)")
    parser.add_argument('--combined-out',dest="combinedout",type=str,default=None,help="filepath for where you would like to store the combined visual plot, both dag and timeline gantt (.pdf/.png)")
    parser.add_argument('--timeline-out',dest="timelineout",type=str,default=None,help="filepath for where you would like to store the timeline gantt (.pdf/.png)")
    parser.add_argument('--dag-out',dest="dagout",type=str,default=None,help="filepath for where you would like to store the dag visual plot (.pdf/.png)")

    parser.add_argument('--title-string',dest="titlestring",type=str,default="",help="the title string for the plot(s)")
    parser.add_argument('--drop',dest="drop",type=str,default=None,help="elements to drop/exclude from the timeline plots")

    args = parser.parse_args()
    timeline_file = args.timeline
    dag_file      = args.dag
    output_file   = args.combinedout
    timeline_output_file  = args.timelineout
    dag_output_file       = args.dagout
    title_string  = args.titlestring
    dropsy = []
    if args.drop is not None:
      dropsy = str(args.drop).split(',')

    if output_file is None and timeline_output_file is None and dag_output_file is None:
        print("Incorrect Arguments. Please provide *at least* one output medium (--combined-out, --timeline-out, --dag-out)")
        sys.exit(1)
    cp = CombinePlots(timeline_file=timeline_file, dag_file=dag_file, combined_output_file=output_file, timeline_output_file=timeline_output_file, dag_output_file=dag_output_file, title_string=title_string, drop=dropsy, use_device_background_colour=True)

