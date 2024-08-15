
class LoadData:
    def __init__(self, directory, custom_policy_rename, subtract_gnn_overhead):
        import os
        import pandas as pandas
        # load in all results
        print("Loading results from: {}".format(directory))
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        data = []
        for f in files:
            if "datamemlinear10" in f: continue #skip the deep-dive result
            if ".gnnoverhead" in f: continue #skip any gnnoverhead files---these will be processed directly
            content = pandas.read_csv(os.path.join(directory, f))
            dag,schedule,system,*_ = f.replace('.csv','').split('-')
            start = min(content['start'])
            end = max(content['end'])
            if subtract_gnn_overhead == True    and "custom" in schedule:
                overhead = open(os.path.join(directory,f.replace('.csv','.gnnoverhead')), "r").read()
                end = end - float(overhead)
            if custom_policy_rename is not None and "custom" in schedule:
                schedule = custom_policy_rename
            data.append({"system-and-policy":"{}-{}".format(system,schedule), "duration":end-start, "dag":dag})
        #if only one system ---just present the policy 
        systems = [d['system-and-policy'].split('-')[0] for d in data]
        self.only_one_system = len(set(systems)) == 1
        if self.only_one_system:
            for d in data:
                d['system-and-policy'] = d['system-and-policy'].split('-')[1]

        self.data = data

    def GetData(self):
        return (self.data)

    def OnlyOneSystem(self):
        return (self.only_one_system)
