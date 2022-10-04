"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)
"""

class Gantt():

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pandas
    from bokeh import palettes

    def _createGanttChart(self, timings, title=None, edgepalette=None, insidepalette=None, time_range=None, zoom=False, drop=[], outline=True, inner_label=False):
        import numpy as np
        timings['taskname'] = np.where(~timings['taskname'].isnull(),timings['taskname'],timings['type'])

        for d in drop:
          timings = timings[timings.taskname != d]

        from natsort import humansorted
        processors = sorted(list(set(timings['acclname'])))

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
        fig = self.plt.figure(figsize=(15,6)) # orig
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
        self.plt.ylabel('Accelerator', fontsize=10)
        self.plt.xlabel('Time (s)', fontsize=10)
        self.plt.setp(labelsy, fontsize = 8)
        ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
        ax.set_xlim(xmin = -0.1, xmax = max(timings['end'])+0.1)

        # sort both labels and handles by labels
        handles, labels = ax.get_legend_handles_labels()
        neworder = dict(humansorted(zip(labels, handles)))
        ax.legend(neworder.values(), neworder.keys(), bbox_to_anchor=(1.15, 1), ncol = 1)

        if time_range and not zoom:
            ax.set_xlim(time_range)
        elif zoom:
            padding = (max(timings['start'])-min(timings['end']))/10 #one-tenth of the total span is at the start and end of the zoomed plot
            ax.set_xlim(xmin=min(timings['start'])-padding,xmax=max(timings['end'])+padding)
        ax.grid(color = 'g', linestyle = ':', alpha=0.5)

        if title:
            self.plt.title(str(title))
        self.plt.tight_layout()

        font = self.font_manager.FontProperties(size='small')

    def showGanttChart(self,timing_log,title=None, **kargs):
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
        self._createGanttChart(timing_content,title, **kargs)
        self.plt.show()

    def saveGanttChart(self,timing_log,file,title=None,**kargs):
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

        self._createGanttChart(timing_content,title,**kargs)
        self.plt.savefig(file)
        #self.plt.savefig(file + '.svg')
        #self.plt.savefig(file + '.pdf')
        #self.plt.savefig(file + '.png')

