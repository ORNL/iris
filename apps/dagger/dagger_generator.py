#!/usr/bin/env python

"""
Adapted from https://github.com/ANRGUSC/Automatic-DAG-Generator and adapted as necessary (to generate IRIS JSON tasks, added kernel-splitting with probabilities etc)
    Authors: Diyi Hu, Jiatong Wang, Quynh Nguyen, Bhaskar Krishnamachari
    Copyright (c) 2018, Autonomous Networks Research Group. All rights reserved.
    license: GPL
"""
__author__ = "Beau Johnston"
__copyright__ = "Copyright (c) 2020-2022, Oak Ridge National Laboratory (ORNL) Programming Systems Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "1.0"

import json
import argparse
import numpy
import random
from bokeh.palettes import brewer
from functools import reduce
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import re
import copy

_kernels = []
_k_probs = []
_depth, _num_tasks, _min_width, _max_width = None,None,None,None
_mean, _std_dev, _skips = None,None,None
_seed = None
_kernel_buffs = {}
_concurrent_kernels = {}
_dimensionality = {}

def parse_args():

    parser = argparse.ArgumentParser(description='DAGGER: Directed Acyclic Graph Generator for Evaluating Runtimes')
    parser.add_argument("--kernels",required=True,type=str,help="The kernel names --in the current directory-- to generate tasks, presented as a comma separated value string e.g. \"process,matmul\"")
    parser.add_argument("--kernel-split",required=False,type=str,help="The percentage of each kernel being assigned to the task, presented as a comma separated value string e.g. \"80,20\".")
    parser.add_argument("--duplicates",required=False,type=int,help="Duplicate the generated DAG horizontally the given number across (to increase concurrency)", default=0)
    parser.add_argument("--concurrent-kernels",required=False,type=str,help="**UNIMPLEMENTED**The number of duplicate/concurrent memory buffers allowed for each kernel, stored as a key value pair, e.g. \"process:2\" indicates that the kernel called \"process\" will only allow two unique sets of memory buffers in the generated DAG, effectively limiting concurrency by indicating a data dependency.",default=None)
    parser.add_argument("--buffers-per-kernel",required=True,type=str,help="The number and type of buffers of buffers required for each kernel, stored as a key value pair, with each buffer separated by white-space, e.g. \"process:r r w rw\" indicates that the kernel called \"process\" requires four separate buffers with read, read, write and read/write permissions respectively")
    parser.add_argument("--kernel-dimensions",required=True,type=str,help="The dimensionality of each kernel, presented as a key-value store, multiple kernels are specified as a comma-separated-value string e.g. \"process:1,matmul:2\". indicates that kernel \"process\" is 1-D while \"matmul\" uses 2-D workgroups.")
    parser.add_argument("--depth",required=True,type=int,help="Depth of tree, e.g. 10.")
    parser.add_argument("--num-tasks",required=True,type=int,help="Total number of tasks to build in the DAG, e.g. 100.")
    parser.add_argument("--min-width",required=True,type=int,help="Minimum width of the DAG, e.g. 1.")
    parser.add_argument("--max-width",required=True,type=int,help="Maximum width of the DAG, e.g. 10.")
    parser.add_argument("--cdf-mean",required=False,type=float,help="Mu of the Cumulative Distribution Function, default=0",default=0)
    parser.add_argument("--cdf-std-dev",required=False,type=float,help="Sigma^2 of the Cumulative Distribution Function, default=0.2",default=0.2)
    parser.add_argument("--skips",required=False,type=int,help="Maximum number of jumps down the DAG levels (Delta) between tasks, default=1",default=1)
    parser.add_argument("--seed",required=False,type=int,help="Seed for the random number generator, default is current system time", default=None)
    parser.add_argument("--sandwich",help="Sandwich the DAG between a lead in and terminating task (akin to a scatter-gather)", action='store_true')

    args = parser.parse_args()

    global _kernels, _k_probs, _depth, _num_tasks, _min_width, _max_width, _mean, _std_dev, _skips, _seed, _sandwich, _concurrent_kernels, _duplicates, _dimensionality

    _depth = args.depth
    _num_tasks = args.num_tasks
    _min_width = args.min_width
    _max_width = args.max_width
    _mean,_std_dev,_skips = args.cdf_mean,args.cdf_std_dev,args.skips
    _seed = args.seed
    _sandwich = args.sandwich
    _duplicates = args.duplicates
    if _duplicates <= 1: _duplicates = 1
    _kernels = args.kernels.split(',')

    #if kernel-split is used ensure they have the same number of percentages as kernels
    if args.kernel_split:
        probs = args.kernel_split.split(',')
        assert len(probs) == len(_kernels), "When specifying --kernel-split there should be as many percentages provided as there are kernels, i.e. all kernels should have a probability of being used in a task"
        #ensure the probabilities add up to 100
        _k_probs = list(map(int,probs))
        assert sum(_k_probs) == 100, "When specifying --kernel-split the percentage of all kernels should add to 100%"
    else:
        _k_probs = numpy.repeat(100/len(_kernels),len(_kernels))
    #process the unique buffers per kernel
    for i in args.buffers_per_kernel.split(','):
        try:
            kernel_name, memory_buffer_permissions = i.split(':')

            if kernel_name not in _kernels:
                raise Exception("{} is not in one of the --kernels.".format(kernel_name))
            memory_buffers = []
            for j in memory_buffer_permissions.split(" "):
                if j == '':
                    continue
                if j != 'r' and j != 'w' and j != 'rw':
                    raise Exception("{} is not of type r, w or rw".format(j))
                memory_buffers.append(j)
            _kernel_buffs[kernel_name] = memory_buffers
        except:
            assert False, "Incorrect arguments given to --buffers-per-kernel. Broken on {}".format(i)
    #parser.add_argument("--buffers-per-kernel",required=True,type=str,help="The number and type of buffers of buffers required for each kernel, stored as a key value pair, with each buffer separated by white-space, e.g. \"process:r r w rw\" indicates that the kernel called \"process\" requires four separate buffers with read, read, write and read/write permissions respectively")

    #process concurrent-kernels
    if args.concurrent_kernels is None:
        for k in _kernels:
            _concurrent_kernels[k] = 0
    else:
        for i in args.concurrent_kernels.split(','):
            try:
                kernel_name, number_of_concurrent_kernels = i.split(':')
                _concurrent_kernels[kernel_name] = int(number_of_concurrent_kernels)
            except:
                assert False, "Incorrect arguments given to --concurrent-kernels. Broken on {}".format(i)
           # \"process:2\" indicates that the kernel called \"process\" will only allow two unique sets of memory buffers in the generated DAG, effectively limiting concurrency by indicating a data dependency.")

    for i in args.kernel_dimensions.split(','):
        try:
            kernel_name, kernel_dimensionality = i.split(':')
            _dimensionality[kernel_name] = int(kernel_dimensionality)
        except:
            assert False, "Incorrect arguments given to --kernel-dimensions. Broken on {}".format(i)
    return

def random_list(depth,total_num,width_min,width_max):
    list_t = []
    if _sandwich: list_t.append(1)
    assert width_min <= total_num , "--num-tasks must be greater than or equal to the --min-width"
    if _sandwich: total_num += 1 # the desired total number has a leading and terminating task which connects all tasks
    for i in range(depth):
        elements_to_add_at_this_level = random.randint(width_min,width_max)
        #if the last level wants to add too many elements
        if sum(list_t) + elements_to_add_at_this_level >= total_num:
            #just add the left overs
            elements_to_add_at_this_level = total_num - sum(list_t)
        list_t.append(elements_to_add_at_this_level)
        if sum(list_t) >= total_num:
            break
    if _sandwich: list_t.append(1)
    return list_t

def gen_task_nodes(depth,total_num,width_min,width_max):
    #num_levels = depth+2		# 2: 1 for entry task, 1 for exit task
    if _sandwich :
        num_list = random_list(depth+2,total_num,width_min,width_max)
    else:
        num_list = random_list(depth,total_num,width_min,width_max)
    num_levels = len(num_list)
    num_nodes_per_level = numpy.array(num_list)
    if _sandwich: num_nodes_per_level[0] = 1.
    if _sandwich: num_nodes_per_level[-1] = 1.

    num_nodes = num_nodes_per_level.sum()
    level_per_task = reduce(lambda a,b:a+b, [[enum]*val for enum,val in enumerate(num_nodes_per_level)],[])
    #e.g. [0,1,2,2,3,3,3,3,4]
    level_per_task = {i:level_per_task[i] for i in range(num_nodes)}
    #level_per_task in the format of {task_i: level_of_i}
    task_per_level = {i:[] for i in range(num_levels)}
    for ti,li in level_per_task.items():
        task_per_level[li] += [ti]
        # task_per_level in the format of {level_i:[tasks_in_level_i]}
    return task_per_level, level_per_task

def gen_task_links(deg_mu,deg_sigma,task_per_level,level_per_task,delta_lvl=2):
    num_tasks = len(level_per_task)
    num_level = len(task_per_level)
    neighs_top_down = {t:numpy.array([]) for t in range(num_tasks)}
    neighs_down_top = {t:numpy.array([]) for t in range(num_tasks)}
    deg = numpy.random.normal(deg_mu,deg_sigma,num_tasks)
    deg2 = (deg/2.).astype(int)
    deg2 = numpy.clip(deg2,1,20)
    #add edges from top to down with deg2, then bottom-up with deg2
    edges = []
    # ---- top-down ----
    for ti in range(num_tasks):
        if level_per_task[ti] == num_level-1:	# exit task is a sink
           continue
        ti_lvl = level_per_task[ti]
        child_pool = []
        for li,tli in task_per_level.items():
            if li <= ti_lvl or li > ti_lvl+delta_lvl:
                continue
            child_pool += tli
        neighs_top_down[ti] = numpy.random.choice(child_pool,min(deg2[ti],len(child_pool)),replace=False)
        edges += [(str(ti),str(ci)) for ci in neighs_top_down[ti]]
    # ---- down-top ----
    for ti in reversed(range(num_tasks)):
        if level_per_task[ti] == 0:
            continue
        ti_lvl = level_per_task[ti]
        child_pool = []
        for li,tli in task_per_level.items():
            if li >= ti_lvl or li < ti_lvl-delta_lvl:
                continue
            child_pool += tli
        neighs_down_top[ti] = numpy.random.choice(child_pool,min(deg2[ti],len(child_pool)),replace=False)
        edges += [(str(ci),str(ti)) for ci in neighs_down_top[ti]]
    return list(set(edges)),neighs_top_down,neighs_down_top

def prune_edges_from_dependencies(task_dag,edges):
    new_edges = []
    for e in edges:
        for t in task_dag:
            if t['name'] == 'task'+e[1] and str('task'+e[0]) in t['depends']:
                new_edges.append(e)
    return new_edges

def repack_dag_with_missing_edges(neighs_down_top,neighs_top_down):
    linked_neighs = neighs_down_top
    for t in neighs_top_down:
        for e in neighs_top_down[t]:
            if t not in linked_neighs[e]:
                linked_neighs[e] = numpy.append(linked_neighs[e],t)
    return linked_neighs

def gen_attr(tasks,kernel_names,kernel_probs):
    #TODO: how do handle memory transfers between each task? Insert h2d and d2h calls around each kernel -- but how should we treat this when multiple dependencies are scheduled on the same device, are they simple dropped in 1024dagger.c?
    #TODO: we should automate kernel arguments being set
    #TODO: handle transfers and arguments
    #TODO: how do we automatically handle user-x?

    #NOTE: TOP-DOWN data-structure (aka tasks in this function) doesn't pin the dependency to how the edges may look--how to determine this, is it sufficient to prune them in the prune_edge function?
    #To perform a weighted selection for task assignment:
    #task_names = random.choices(kernel_names, weights=kernel_probs, k=len(tasks))

    #set of kernel names to use based on the probabilities
    bag_of_kernels = random.choices(kernel_names, weights=kernel_probs, k=len(tasks))
    
    dag = []
    for i in range(0,len(tasks)):
        tname = "task"+str(i)

        kname = bag_of_kernels.pop()
        #selected_memory = random.choices(range(0,_concurrent_kernels[kname]),k=1)
        selected_memory = [0]
        kinst = int(selected_memory[0])

        deps  = tasks[i]
        deps = ["task"+str(d) for d in deps]

        #for ck in range(0,_concurrent_kernels[kname]):
        mobs = []
        mper = []

        for i,j in enumerate(_kernel_buffs[kname]):
            mobs.append("devicemem-{}-buffer{}-instance{}".format(kname,i,kinst))
            mper.append(j)

        if mobs == []:
            assert False, "kernel {} hasn't been given any memory to work on! This can be rectified by setting --buffers-per-kernel.".format(kname)
        #TODO the dimensionality of each work group can also be set instead of a fixed "user-size":
        kernel_dimension = []
        for i in range(0,_dimensionality[kname]):
            kernel_dimension.append("user-size")
        #task_instance_name = tname
        #if ck != 0: #if we're supporting concurrent tasks, all but the first instance should have a unique name
        #    task_instance_name += '-' + str(ck)
        #TODO may need to undo the following:
        parameters = []
        for i,m in enumerate(mobs):
          parameters.append({"type":"memory_object","name":m,"permission":mper[i]})
        #for m in mper:
        #  parameters.append({"type":"scalar","name":m})
        task = {"name":tname,"commands":[{"kernel":{"name":kname,"global_size":kernel_dimension,"parameters":parameters}}],"depends":deps,"target":"user-target-data"}
        #task = {"name":tname,"commands":[{"kernel":kname,["user-size"],mobs,mper]}],"depends":deps,"target":"user-target-data"}
        #was:
        #task = {"name":tname,"kernel":[kname,"user-size",["user-mem"],["rw"] ],"depends":deps,"target":"user-target"}
        dag.append(task)

    return dag

def duplicate_for_concurrency(task_dag,edges):
    if _duplicates <= 1:
        return task_dag,edges
    #the standard structure has been generated now duplicate for the number of concurrent kernels
    new_dag   = copy.deepcopy(task_dag)
    new_edges = copy.deepcopy(edges)
    ckernels  = _concurrent_kernels

    #get the maximally concurrent kernel
    #max_ckernel = None
    #for key,value in ckernels.items():
    #    if value == max(ckernels.values()):
    #        max_ckernel = key
    #assert max_ckernel != None

    #get the last task name --- this indicates the number of tasks in each chain
    original_chain_length = len(task_dag)

    #first sweep and generate the broad structure
    for c in range(1, _duplicates):#the maximum degree of concurrency indicates how many "parallel" DAGs to paste in
        y = original_chain_length*c
        for chain_index in range(0,original_chain_length):
            x = copy.deepcopy(task_dag[chain_index])
            #this is a concurrent instance!
            x['name'] = "task"+str(y)
            #update the instance buffers used in this concurrent instance
            for zi, z in enumerate(x['commands'][0]['kernel']['parameters']):
                old_instance_buffer_name = re.findall(r'(.*instance)\d+$',z['name'])
                old_instance_buffer_name = old_instance_buffer_name[0]
                nib = (old_instance_buffer_name + str(c))
                x['commands'][0]['kernel']['parameters'][zi]['name'] = nib
            #*ASSUMPTION:* all new dependencies should share the same offset as elements in the chain
            new_dependencies = []
            for z in x['depends']:
                dependency_no = original_chain_length*c+int(z.replace('task',''))
                new_dependencies.append("task"+str(dependency_no))
            x['depends'] = new_dependencies
            y += 1 #increment the new task name counter
            #what about other (non-maximal) concurrency kernels?
            #TODO
            new_dag.append(x)

    #and update edges accordingly
    for c in range(0, _duplicates):#the maximum degree of concurrency indicates how many "parallel" DAGs to paste in
        for entry in edges:
            new_edges.append((str(int(entry[0])+original_chain_length*c),str(int(entry[1])+original_chain_length*c)))

    print("TODO: still need to support concurrent kernels...")

    #update edges
    # then assigned shared / common tasks (tasks with less concurrency)
    #TODO: stitch together shared tasks

    return new_dag,new_edges

def plot_dag(task_dag,edges,dag_path_plot):
    #edge_d = [(e[0],e[1],{'data':neighs_down_top[i]}) for i,e in enumerate(edges)]
    edge_d = [(e[0],e[1],{"depends":task_dag[int(e[0])]['name']}) for i, e in enumerate(edges)]
    #edge_d = [(e[i][], e['name'],{'data':e['name']}) for i,e in enumerate(task_dag)]

    dag = nx.DiGraph()
        #hash the colours in the dag for each kernel name
    unique_kernels = sorted(set([t['commands'][0]['kernel']['name'] for t in task_dag]))
    assert len(unique_kernels) < 9, "Can't colourize that many unique kernels -- maybe choose a bigger colour palette? <https://docs.bokeh.org/en/latest/docs/reference/palettes.html>"
    palette = brewer['Pastel1'][9]
    phash = {}
    for i,k in enumerate(unique_kernels):
        phash[k] = palette[i]

    #if there are edges, create the DAG from it, otherwise just the nodes (there are no edges in the DAG, and so the draw call will fail)
    node_d = [(str(i),{"label":e['name'], "position":(i,0)}) for i, e in enumerate(task_dag)]
    dag.add_nodes_from(node_d)
    node_labels = {str(i):'{}'.format(n['name']) for i,n in enumerate(task_dag)}
    if edge_d != []:
        dag.add_edges_from(edge_d)
        edge_labels = {(e1,e2):'{}'.format(d) for e1,e2,d in dag.edges(data=True)}

    #plot it
    pos = graphviz_layout(dag,prog='dot')
    #add colour
    node_colours = ['{}'.format(phash[task_dag[int(n)]['commands'][0]['kernel']['name']]) for n in dag]

    nx.draw(dag,pos=pos,labels=node_labels,font_size=8,node_color=node_colours)
    #nx.draw_networkx_edge_labels(dag,pos,edge_labels=edge_labels,label_pos=0.75,font_size=6)
    plt.savefig(dag_path_plot)


#NOTE: the order the kernels are provided by --kernels, impacts on the order the input buffers are included
def determine_iris_inputs():
    inputs = []
    print('TODO: could handle multiple dimensions of execution here')
    inputs.append("user-size")
    for k in _kernels:
        inputs.append("user-size-cb-{}".format(k))
    for k in _kernels:
        for ck in range(0,_duplicates):
            for i,j in enumerate(_kernel_buffs[k]):
                #inputs.append("hostmem-{}-buffer{}".format(k,i))
                #TODO could facilitate concurrent-kernels with :
                inputs.append("hostmem-{}-buffer{}-instance{}".format(k,i,ck))
    for k in _kernels:
        for ck in range(0,_duplicates):
            for i,j in enumerate(_kernel_buffs[k]):
                inputs.append("devicemem-{}-buffer{}-instance{}".format(k,i,ck))
    print('TODO: support multiple targets--hint all but the h2d and d2h memory transfers should be transfer target (control-dependency) while all actual kernels should use the data-dependency')
    inputs.append("user-target-control")
    inputs.append("user-target-data")
    #was "inputs":[ "user-size", "user-size-cb", "user-A", "user-B", "user-mem", "user-target" ]
    return inputs

def determine_and_prepend_iris_h2d_transfers(dag):
    #sample:       {
    #      "name" : "transferto0",
    #      "h2d": ["user-memA0", "user-A", "0", "user-size-cb"],
    #      "h2d": ["user-memB0", "user-B", "0", "user-size-cb"],
    #      "target": "user-target1"
    #  },
    transfers = []
    #this should be extended for each instance
    for i,k in enumerate(_kernels):
        for ck in range(0,_duplicates):
            for j,l in enumerate(_kernel_buffs[k]):
                if l == 'w': #Only transfer memory that will be read on the device
                    continue
                transfer = {}
                transfer["name"] = "transferto-{}-buffer{}-instance{}".format(k,j,ck)
                buffer_name = "devicemem-{}-buffer{}-instance{}".format(k,j,ck)
                commands = {}
                commands["h2d"] = {"name":"h2d-buffer{}-instance{}".format(j,ck),"device_memory":buffer_name,"host_memory":"hostmem-{}-buffer{}-instance{}".format(k,j,ck),"offset":"0","size":"user-size-cb-{}".format(k)}
                #commands["h2d"] = buffer_name, "hostmem-{}-buffer{}-instance{}".format(k,j,ck), "0", "user-size-cb-{}".format(k)
                transfer["commands"] = [commands]
                #transfer["h2d"] = [buffer_name, "hostmem-{}-buffer{}-instance{}".format(k,j,ck), "0", "user-size-cb-{}".format(k)]
                transfer["target"] = "user-target-control"
                memory_instance_in_use = False
                transfer["depends"] = []
                #add as a dependency for this kernel
                for m,t in enumerate(dag):
                    if 'kernel' not in t['commands'][0]:
                        continue
                    for p in t['commands'][0]['kernel']['parameters']:
                      if buffer_name == p['name']:
                          memory_instance_in_use = True
                          #TODO: sort out concurrency
                          if transfer["name"] not in dag[m]['depends']:
                              dag[m]['depends'].append(transfer["name"])
                #only include memory transfers for memory which is actually in use
                if memory_instance_in_use == True:
                    transfers.append(transfer)

    #prepend the h2d transfers
    for t in range(0, len(transfers)):
        dag.insert(0,transfers.pop())
    return dag

def determine_and_append_iris_d2h_transfers(dag):
    transfers = []
    #this should be extended for each instance
    for i,k in enumerate(_kernels):
        for ck in range(0,_duplicates):
            for j,l in enumerate(_kernel_buffs[k]):
                if l == 'r': #Only transfer memory that have been written on the device
                    continue
                transfer = {}
                transfer["name"] = "transferfrom-{}-buffer{}-instance{}".format(k,j,ck)
                buffer_name = "devicemem-{}-buffer{}-instance{}".format(k,j,ck)
                commands = {}
                commands["d2h"] = {"name":"d2h-buffer{}-instance{}".format(j,ck),"device_memory":buffer_name,"host_memory":"hostmem-{}-buffer{}-instance{}".format(k,j,ck),"offset":"0","size":"user-size-cb-{}".format(k)}
                #commands["d2h"] = buffer_name, "hostmem-{}-buffer{}-instance{}".format(k,j,ck), "0", "user-size-cb-{}".format(k)
                transfer["commands"] = [commands]
                #transfer["d2h"] = [buffer_name, "hostmem-{}-buffer{}-instance{}".format(k,j,ck), "0", "user-size-cb-{}".format(k)]
                transfer["depends"] = []
                #add this dependency to the DAGs--depend on all kernels which use these buffers
                memory_instance_in_use = False
                for m,t in enumerate(dag):
                    if 'kernel' not in t['commands'][0]:
                        continue
                    for p in t['commands'][0]['kernel']['parameters']:
                      if buffer_name == p['name']:
                          memory_instance_in_use = True
                          transfer["depends"].append(t["name"])
                transfer["target"] = "user-target-control"
                if memory_instance_in_use == True:
                    transfers.append(transfer)

    #prepend the d2h transfers
    for t in range(0, len(transfers)):
        dag.append(transfers.pop(0))
    return dag

def get_task_to_json(dag,deps):
    #add dependencies to the dag structure
    for t in deps:
        #print("task"+t[1]+" depends on task"+t[0])
        #print("adding dependency to "+dag[int(t[1])]['name'])
        dag[int(t[1])]['depends'].append("task"+t[0])
    #remove duplicate dependencies
    for t in dag:
        t['depends']=list(dict.fromkeys(t['depends']))
        #print(t['depends'])
    f = open("graph.json", 'w')
    inputs = determine_iris_inputs()
    dag = determine_and_prepend_iris_h2d_transfers(dag)
    dag = determine_and_append_iris_d2h_transfers(dag)
    final_json = {"iris-graph":{"inputs":inputs,"graph":{"tasks":dag}}}
    f.write(json.dumps(final_json,indent = 2))
    f.close()

if __name__ == '__main__':
    parse_args()
    random.seed(_seed)
    task_per_level, level_per_task = gen_task_nodes(_depth,_num_tasks,_min_width,_max_width)
    edges,neighs_top_down,neighs_down_top = gen_task_links(_mean,_std_dev,task_per_level,level_per_task,delta_lvl=_skips)
    if _sandwich:
        neighs_down_top = repack_dag_with_missing_edges(neighs_down_top,neighs_top_down)
    task_dag = gen_attr(neighs_down_top,_kernels,_k_probs)
    edges = prune_edges_from_dependencies(task_dag,edges)
    task_dag,edges = duplicate_for_concurrency(task_dag,edges)
    #print("task_dag:")
    #print(task_dag)
    #print("\n\n")
    #print("edges:")
    #print(edges)

    plot_dag(task_dag,edges,'./dag.png')
    #dag = get_task_dag(args.conf,"dag.png")

    #get_task_to_dag(dag)
    #get_task_to_communication(dag)
    #get_task_to_dummy_app()
    #get_task_to_generate_file(dag)
    get_task_to_json(task_dag,edges)
    print('done')
