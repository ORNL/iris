#!/usr/bin/env python

"""
Adapted from https://github.com/ANRGUSC/Automatic-DAG-Generator and adapted as necessary (to generate IRIS JSON tasks, added kernel-splitting with probabilities etc)
    Authors: Diyi Hu, Jiatong Wang, Quynh Nguyen, Bhaskar Krishnamachari
    Copyright (c) 2018, Autonomous Networks Research Group. All rights reserved.
    license: GPL
"""
__author__ = "Beau Johnston"
__copyright__ = "Copyright (c) 2020-2023, Oak Ridge National Laboratory (ORNL) Programming Systems Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "1.0"
__schema__ = "https://raw.githubusercontent.com/ORNL/iris/v2.0.0/schema/dagger.schema.json"

import json
import argparse
import numpy
import random
import shlex
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
_total_num_concurrent_kernels = 0
_dimensionality = {}
_graph = None
_memory_object_pool = None
_local_sizes = {}
_memory_shuffle_count = 0

def init_parser(parser):
    parser.add_argument("--graph",default="graph.json",type=str,help="The DAGGER json graph file to load, e.g. \"graph.json\".")
    parser.add_argument("--kernels",required=True,type=str,help="The kernel names --in the current directory-- to generate tasks, presented as a comma separated value string e.g. \"process,matmul\".")
    parser.add_argument("--kernel-split",required=False,type=str,help="The percentage of each kernel being assigned to the task, presented as a comma separated value string e.g. \"80,20\".")
    parser.add_argument("--duplicates",required=False,type=int,help="Duplicate the generated DAG horizontally the given number across (to increase concurrency).", default=0)
    parser.add_argument("--concurrent-kernels",required=False,type=str,help="The number of concurrent memory buffers allowed for each kernel, stored as a key value pair, e.g. \"process:2\" indicates that the kernel called \"process\" will only allow two unique sets of memory buffers in the generated DAG, effectively limiting concurrency by indicating a data dependency.",default=None)
    parser.add_argument("--buffers-per-kernel",required=True,type=str,help="The number and type of buffers of buffers required for each kernel, stored as a key value pair, with each buffer separated by white-space, e.g. \"process:r r w rw\" indicates that the kernel called \"process\" requires four separate buffers with read, read, write and read/write permissions respectively.")
    parser.add_argument("--kernel-dimensions",required=True,type=str,help="The dimensionality of each kernel, presented as a key-value store, multiple kernels are specified as a comma-separated-value string e.g. \"process:1,matmul:2\". indicates that kernel \"process\" is 1-D while \"matmul\" uses 2-D workgroups.")
    parser.add_argument("--local-sizes",required=False,type=str,help="The local-workgroup size of each kernel, presented as a key-value store, multiple kernels are specified as a comma-separated-value string e.g. \"process:128 128,matmul:256 1 1\".")
    parser.add_argument("--depth",required=True,type=int,help="Depth of tree, e.g. 10.")
    parser.add_argument("--num-tasks",required=True,type=int,help="Total number of tasks to build in the DAG, e.g. 100.")
    parser.add_argument("--min-width",required=True,type=int,help="Minimum width of the DAG, e.g. 1.")
    parser.add_argument("--max-width",required=True,type=int,help="Maximum width of the DAG, e.g. 10.")
    parser.add_argument("--cdf-mean",required=False,type=float,help="Mu of the Cumulative Distribution Function, default=0.",default=0)
    parser.add_argument("--cdf-std-dev",required=False,type=float,help="Sigma^2 of the Cumulative Distribution Function, default=0.2.",default=0.2)
    parser.add_argument("--skips",required=False,type=int,help="Maximum number of jumps down the DAG levels (Delta) between tasks, default=1.",default=1)
    parser.add_argument("--seed",required=False,type=int,help="Seed for the random number generator, default is current system time.", default=None)
    parser.add_argument("--sandwich",help="Sandwich the DAG between a lead in and terminating task (akin to a scatter-gather).", action='store_true')
    parser.add_argument("--num-memory-objects",required=False,type=int,help="Enables sharing of memory objects dealt between tasks! It is the total number of memory objects to be passed around in the DAG between tasks (allows greater task interactions).", default=None)
    parser.add_argument("--use-data-memory",required=False,help="Enables the graph to use memory instead of the default explicit memory buffers. This results in final explicit flush events of buffers that are written.",default=False,action='store_true')
    parser.add_argument("--num-memory-shuffles",required=False,type=int,help="Memory shuffles is the number of swaps (positive integer) between memory buffers based on a random task selection and random memory buffer selection.",default=0)
    parser.add_argument("--handover-in-memory-shuffle",required=False,help="Setting this argument to True yields a stricter form of Memory shuffle where each swap ensures the selected memory buffer is of different permissions, this results in the same memory being written to in one kernel then later read from.",default=False,action='store_true')
    parser.add_argument("--no-deps",required=False,help="Disable explicit task dependencies -> rely on IRIS's data-flow dependency tracking.",default=False,action='store_true')
    parser.add_argument("--no-flush",required=False,help="Disable explicit data flushing -> rely on IRIS's data-flow dependency tracking.",default=False,action='store_true')


def parse_args(pargs=None,additional_arguments=[]):
    parser = argparse.ArgumentParser(description='DAGGER: Directed Acyclic Graph Generator for Evaluating Runtimes')

    init_parser(parser)

    # Allow other arguments to be added.
    for init in additional_arguments:
        init(parser)

    if pargs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(pargs))

    global _kernels, _k_probs, _depth, _num_tasks, _min_width, _max_width, _mean, _std_dev, _skips, _seed, _sandwich, _concurrent_kernels, _duplicates, _dimensionality, _graph, _memory_object_pool, _use_data_memory, _total_num_concurrent_kernels, _local_sizes, _memory_shuffle_count, _handover, _no_deps, _no_flush

    _graph = args.graph
    _depth = args.depth
    _num_tasks = args.num_tasks
    _min_width = args.min_width
    _max_width = args.max_width
    _mean,_std_dev,_skips = args.cdf_mean,args.cdf_std_dev,args.skips
    _seed = args.seed
    _sandwich = args.sandwich
    _duplicates = args.duplicates
    _memory_shuffle_count = args.num_memory_shuffles
    _handover = args.handover_in_memory_shuffle
    _no_deps = args.no_deps
    _no_flush = args.no_flush

    if _duplicates <= 1: _duplicates = 1

    _kernels = args.kernels.split(',')
    _use_data_memory = args.use_data_memory

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
                if "scalar" in j:
                    scalar_type,value,data_type,data_size = j.split(";");
                    # and convert to the appropriate type
                    data_size = int(data_size)
                    if data_type == "int":
                        value = int(value)
                    if data_type == "float":
                        value = float(value)
                    if data_type == "short":
                        value = int(value)
                    if data_type == "long":
                        value = int(value)
                    if data_type == "double":
                        value = float(value)
                    memory_buffers.append({"type":scalar_type,
                        "value": value,
                        "data_type":data_type,
                        "data_size":data_size})
                    continue
                elif j != 'r' and j != 'w' and j != 'rw':
                    raise Exception("{} is not of type r, w or rw".format(j))
                memory_buffers.append(j)
            _kernel_buffs[kernel_name] = memory_buffers
        except:
            assert False, "Incorrect arguments given to --buffers-per-kernel. Broken on {}".format(i)
    if args.num_memory_objects is not None:
        num_provided_memory_objects = 0
        for k in _kernel_buffs:
            num_provided_memory_objects += len(_kernel_buffs[k])
        assert args.num_memory_objects < num_provided_memory_objects, "Incorrect arguments given to --num-memory-objects. The number should be less than the default {} . Broken on {}".format(num_provided_memory_objects,i)
        _memory_object_pool = []
        n = 0
        for kname in _kernel_buffs:
            kinstk = 0
            for i,j in enumerate(_kernel_buffs[kname]):
                if n >= args.num_memory_objects:
                    break
                #"devicemem-{}-buffer{}-instance{}".format(kname,i,kinst)
                _memory_object_pool.append("devicemem-{}-buffer{}-instance{}".format(kname,i,kinstk))
                n += 1

    #process concurrent-kernels
    if args.concurrent_kernels is None:
        for k in _kernels:
            _concurrent_kernels[k] = 1
            _total_num_concurrent_kernels  += _concurrent_kernels[k]
    else:
        for i in args.concurrent_kernels.split(','):
            try:
                kernel_name, number_of_concurrent_kernels = i.split(':')
                _concurrent_kernels[kernel_name] = int(number_of_concurrent_kernels)
                _total_num_concurrent_kernels  += int(number_of_concurrent_kernels)
            except:
                assert False, "Incorrect arguments given to --concurrent-kernels. Broken on {}".format(i)
           # \"process:2\" indicates that the kernel called \"process\" will only allow two unique sets of memory buffers in the generated DAG, effectively limiting concurrency by indicating a data dependency.")
    #assert _total_num_concurrent_kernels == _max_width, "Incorrect arguments given: the total number of concurrent kernels must be equal to the maximum width of any level in the graph. The total number of concurrent kernels are: {} while the width is {}\n".format(_total_num_concurrent_kernels,_max_width)

    for i in args.kernel_dimensions.split(','):
        try:
            kernel_name, kernel_dimensionality = i.split(':')
            _dimensionality[kernel_name] = int(kernel_dimensionality)
        except:
            assert False, "Incorrect arguments given to --kernel-dimensions. Broken on {}".format(i)

    #process local-workgroup-sizes
    #MARK can be used for offset and gws?
    if args.local_sizes is not None:
        for i in args.local_sizes.split(','):
            try:
                kernel_name, dims = i.split(':')
                _local_sizes[kernel_name] = [ int(x) for x in dims.split(' ') ]
            except:
                assert False, "Incorrect arguments given to --local-sizes. Broken on {}".format(i)
    return args

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
            if t['name'] == "initial_h2d" or t['name'] == "final_d2h" or (t['name'] == 'task'+e[1] and str('task'+e[0]) in t['depends']):
                new_edges.append(e)
    return new_edges

def repack_dag_with_missing_edges(neighs_down_top,neighs_top_down):
    linked_neighs = neighs_down_top
    for t in neighs_top_down:
        for e in neighs_top_down[t]:
            if t not in linked_neighs[e]:
                linked_neighs[e] = numpy.append(linked_neighs[e],t)
    return linked_neighs

def create_h2d_task_from_kernel_bag(kernel_bag):
    #sample:       {
    #      "name" : "transferto0",
    #      "h2d": ["user-memA0", "user-A", "0", "user-size-cb"],
    #      "h2d": ["user-memB0", "user-B", "0", "user-size-cb"],
    #      "target": "user-target1"
    #  },
    commands = []
    for kernel_name in kernel_bag.keys():
        for kernel_instance in kernel_bag[kernel_name]:
            for parameter in kernel_instance['kernel']['parameters']:
                if parameter['type'] == "memory_object":
                    memory_name = parameter['value']
                    size_bytes = parameter['size_bytes']
                    commands.append({ "h2d":{ "name":memory_name.replace("devicemem","transferto"), "device_memory":memory_name, "host_memory":memory_name.replace("devicemem","hostmem"), "offset":"0", "size":size_bytes } })

    transfer_task = { "name":"initial_h2d", "commands":commands, "target":"user-target-control", "depends":[] }
    return transfer_task

def create_d2h_task_from_kernel_bag(_kernel_bag,dag):
    depends = []
    for t in dag:
        depends.append(t['name'])
    touched_memory_objects = []
    size_bytes_lookup = {}
    for task in dag:
        for cmd in task['commands']:
            if 'kernel' in cmd:
                for params in task['commands'][0]['kernel']['parameters']:
                    if params['type'] == 'memory_object' and "w" in params['permissions']:
                        touched_memory_objects.append(params['value'])
                        size_bytes_lookup[params['value']] = params['size_bytes']
    touched_memory_objects = set(touched_memory_objects)
    commands = []
    for memory_name in touched_memory_objects:
        if _use_data_memory:
            commands.append({ "d2h":{ "name":memory_name.replace("devicemem","dmemflush"), "device_memory":memory_name, "data-memory-flush":1 } })
        else:
            commands.append({ "d2h":{ "name":memory_name.replace("devicemem","transferfrom"), "device_memory":memory_name, "host_memory":memory_name.replace("devicemem","hostmem"), "offset":"0", "size":size_bytes_lookup[memory_name] } })
        if _no_flush: commands.pop()
    #remove the initial_h2d as a dependency
    while('initial_h2d' in depends):
        depends.remove('initial_h2d')
    transfer_task = { "name":"final_d2h", "commands":commands, "target":"user-target-control", "depends":depends }
    return transfer_task

def generate_attributes(task_dependencies,tasks_per_level):
    global _memory_shuffle_count, _handover
    #generate internal kernel instance names based on duplicates
    _kernel_bag = {}
    for k in _kernels:
        _kernel_bag[k] = []
        for c in range(1,_concurrent_kernels[k]+1):
            x = {}
            x["name"] = k
            if k in _local_sizes.keys():
                x["local_size"] = _local_sizes[k]

            x["global_size"] = []
            for z in range(0,_dimensionality[k]):
                x["global_size"].append("user-size")

            x["parameters"] = []
            for index,permission in enumerate(_kernel_buffs[k]):
                #if a scalar argument has been set, just pass it on
                if type(permission) is dict and permission['type'] == 'scalar':
                    x["parameters"].append(permission)
                    continue
                x["parameters"].append({
                    "type":"memory_object",
                    "value":"devicemem-{}-buffer{}-instance{}".format(k,index,c),
                    "size_bytes":"user-size-cb-{}".format(k),
                    "permissions":permission})
            _kernel_bag[k].append({"kernel":x})

    dag = []
    current_task_number = 0
    for level_num in tasks_per_level:
        #if sandwich mode ensure the first and last tasks do the h2d and d2h transfers
        if _sandwich:
            #Note: if dmem is used only the final flush is needed, and so the first task is replaced with an arbitrary kernel task
            if current_task_number == 0 and not _use_data_memory:
                dag.append(create_h2d_task_from_kernel_bag(_kernel_bag))
                current_task_number += 1
                continue
            elif current_task_number == _num_tasks + 1:
                dag.append(create_d2h_task_from_kernel_bag(_kernel_bag,dag))
                current_task_number += 1
                print("replacing the last task")
                continue
        #TODO: duplication should work again now --- bring it back, or at least test it out
        tasks_this_level = tasks_per_level[level_num]
        #flatten the _kernel_bag to yield a flattened list to help with the kernel selection
        pool = []
        for k in [_kernel_bag[k] for k in _kernel_bag.keys()]:
            pool.extend(k)
        #if we only have enough tasks as their are task instances then ignore the probabilities of each kernel being used---we've specified the exact instances instead
        if len(tasks_this_level) <= len(pool):
            selected_tasks_this_level = random.sample(pool, k=len(tasks_this_level))
            for selected_task in selected_tasks_this_level:
                deps = ["task"+str(d) for d in task_dependencies[current_task_number]]
                if _no_deps: deps = []
                task = {"name":"task"+str(current_task_number), "commands":[selected_task], "depends":deps, "target":"user-target-data"}
                dag.append(task)
                current_task_number += 1
        #default to the original probabilities of each kernel
        else:
            selected_kernels_this_level = random.choices(_kernels, weights=_k_probs, k=len(tasks_this_level))
            selected_tasks_this_level = []
            for unique_kernel in set(selected_kernels_this_level):
                num_memory_instances = selected_kernels_this_level.count(unique_kernel)
                selected_tasks_this_level = random.sample(_kernel_bag[unique_kernel],k = num_memory_instances)
            for selected_task in selected_tasks_this_level:
                deps = ["task"+str(d) for d in task_dependencies[current_task_number]]
                if _no_deps: deps = []
                task = {"name":"task"+str(current_task_number), "commands":[selected_task], "depends":deps, "target":"user-target-data"}
                dag.append(task)
                current_task_number += 1

    # Memory Shuffle Algorithm:
    # -------------------------
    # input variables: memory_shuffles is the number of swaps to make -- positive integer
    #0) for each memory_shuffle_count
    #1) perform random selection of task and memory buffer
    #2) look for another task with memory buffer
    #3) ensure the we aren't swapping the same buffer
    #4) ensure the selected memory buffer not currently being used in the task (we don't want the kernel reading and writing to the same buffer since that's undefined and dangerous)
    #5) if handover (a stricter form of memory swap) ensure the opposite permission (if the first selected buffer was read only, swap with a write buffer) --- akin to a producer/consumer model
    #6) perform swap
    #7) decrement the memory_shuffle_count
    while _memory_shuffle_count > 0: #0)
        #1)
        selected_tasks = random.sample(range(len(dag)),k=2)
        assert dag[selected_tasks[0]]['commands'][0].keys() == {"kernel"} and dag[selected_tasks[1]]['commands'][0].keys() == {"kernel"}, print("Error: all tasks selected for the memory shuffle should be kernel tasks")
        #2)
        first = dag[selected_tasks[0]]['commands'][0]['kernel']['parameters'] #all kernel tasks *currently* only contain a single command
        second = dag[selected_tasks[1]]['commands'][0]['kernel']['parameters']
        first_mobj = random.choice(range(len(first)))
        second_mobj = random.choice(range(len(second)))
        #3)
        if first[first_mobj]['value'] == second[second_mobj]['value']:
            continue
        #4)
        if second[second_mobj]['value'] in [x['value'] for x in first]:
            continue
        #5)
        if _handover and first[first_mobj]['permissions'] == second[second_mobj]['permissions']:
            continue
        #6)
        print("replacing task: {} arg: {} buffer: {} with buffer: {}".format(dag[selected_tasks[0]]["name"],first_mobj,first[first_mobj]['value'], second[second_mobj]['value']))
        dag[selected_tasks[0]]['commands'][0]['kernel']['parameters'][first_mobj]['value'] = second[second_mobj]['value']
        dag[selected_tasks[1]]['commands'][0]['kernel']['parameters'][second_mobj]['value'] = first[first_mobj]['value']
        #7)
        _memory_shuffle_count -= 1
    return dag

def rename_special_tasks(task_dag):
    dependency_tasks = []
    for t in task_dag:
        dependency_tasks += t['depends']
    dependency_tasks = set(dependency_tasks)
    task_names = []
    for t in task_dag:
        task_names.append(t['name'])
    for t in dependency_tasks:
        if t not in task_names:
            if t == "task0": #renaming task0 to initial_h2d
                #sweep and replace
                for s in task_dag:
                    if t in s['depends']:
                        s['depends'] = [sub.replace(t, 'initial_h2d') for sub in s['depends']]
            else:
                print("found a new task with a unique name {}, don't know what this should be renamed to".format(t))
                import ipdb
                ipdb.set_trace()

    return(task_dag)

def gen_attr(tasks,kernel_names,kernel_probs):
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
        selected_memory = random.choices(range(0,_concurrent_kernels[kname]),k=1)
        print("selected memory {}\n".format(selected_memory))
        #selected_memory = [0]
        kinst = int(selected_memory[0])

        mobs = []
        mper = []

        #time to use the new experimental shared memory objects to interact between tasks
        if _memory_object_pool is not None:
            samp = random.sample(_memory_object_pool,k=len(_kernel_buffs[kname]))
            for j,s in enumerate(samp):
                mobs.append(s)
                #TODO: some magic here based on random.choices for memory object names
                #import ipdb
                #ipdb.set_trace()
                #print("figuring out proper permissions")
                mper.append(_kernel_buffs[kname][j])
        else:
            for z,j in enumerate(_kernel_buffs[kname]):
                mobs.append("devicemem-{}-buffer{}-instance{}".format(kname,z,kinst))
                mper.append(j)

        if mobs == []:
            assert False, "kernel {} hasn't been given any memory to work on! This can be rectified by setting --buffers-per-kernel.".format(kname)
        #TODO the dimensionality of each work group can also be set instead of a fixed "user-size":
        kernel_dimension = []
        for z in range(0,_dimensionality[kname]):
            kernel_dimension.append("user-size")
        #task_instance_name = tname
        #if ck != 0: #if we're supporting concurrent tasks, all but the first instance should have a unique name
        #    task_instance_name += '-' + str(ck)
        parameters = []
        for z,m in enumerate(mobs):
            parameters.append({"type":"memory_object","value":m,"size_bytes":"user-size-cb-{}".format(kname),"permissions":mper[z]})

        #filter out dependencies based on tasks that have shared memory objects
        deps = []
        for t in tasks:
            if t >= i:#we can only insert a dependency on tasks using the same memory object if they were assigned before the current task
                break
            if dag == []:
                continue
            for p in dag[t]['commands'][0]['kernel']['parameters']:
                if p['value'] in mobs:
                    deps.append(t)
                break
        deps = ["task"+str(d) for d in deps]

        task = {"name":tname,"commands":[{"kernel":{"name":kname,"global_size":kernel_dimension,"parameters":parameters}}],"depends":deps,"target":"user-target-data"}
        dag.append(task)

    return dag

def duplicate_for_concurrency(task_dag,edges):
    if _duplicates <= 1:
        return task_dag,edges
    #the standard structure has been generated now duplicate for the number of concurrent kernels
    new_dag   = copy.deepcopy(task_dag)
    new_edges = copy.deepcopy(edges)
    ckernels  = _concurrent_kernels

    #TODO: make more concurrent memory instances based of the duplicates value!

    #get the last task name --- this indicates the number of tasks in each chain
    original_chain_length = len(task_dag)

    #first sweep and generate the broad structure
    for d in range(1, _duplicates):#the number of duplicates indicates how many "parallel" DAGs to paste in
        y = original_chain_length*d
        for chain_index in range(0,original_chain_length):
            x = copy.deepcopy(task_dag[chain_index])
            if "d2h" in task_dag[chain_index]['commands'][0].keys():
                #replace used memory buffers after new ones are added for the new sub-dag series
                x['name'] = x['name']+"-task-"+str(y)
                for z in x['commands']:
                    o,otn = z['d2h']['name'].split('instance')
                    o = o+"instance"
                    otn = str(int(otn)+d)
                    z['d2h']['name'] = "d2h"+str(y)+"-"+o+otn
                    o,otn = z['d2h']['device_memory'].split('instance')
                    o = o+"instance"
                    otn = str(int(otn)+d)
                    z['d2h']['device_memory'] = o+otn
                    if 'data-memory-flush' in z['d2h'].keys():
                        continue
                    o,otn = z['d2h']['host_memory'].split('instance')
                    o = o+"instance"
                    otn = str(int(otn)+d)
                    z['d2h']['host_memory'] = o+otn
            elif "h2d" in task_dag[chain_index]['commands'][0].keys():
                #replace used memory buffers after new ones are added for the new sub-dag series
                x['name'] = x['name']+"-task"+str(d)
                for z in x['commands']:
                    o,otn = z['h2d']['name'].split('instance')
                    o = o+"instance"
                    otn = str(int(otn)+d)
                    z['h2d']['name'] = "h2d"+str(y)+"-"+o+otn
                    o,otn = z['h2d']['device_memory'].split('instance')
                    o = o+"instance"
                    otn = str(int(otn)+d)
                    z['h2d']['device_memory'] = o+otn
                    o,otn = z['h2d']['host_memory'].split('instance')
                    o = o+"instance"
                    otn = str(int(otn)+d)
                    z['h2d']['host_memory'] = o+otn
            else:
                #kernel
                x['name'] = "task"+str(y)
                #update the instance buffers used in this concurrent instance
                for zi, z in enumerate(x['commands'][0]['kernel']['parameters']):
                    old_instance_buffer_name = re.findall(r'(.*instance)\d+$',z['value'])
                    old_instance_buffer_name = old_instance_buffer_name[0]
                    nib = (old_instance_buffer_name + str(d))
                    x['commands'][0]['kernel']['parameters'][zi]['value'] = nib
                    x['commands'][0]['kernel']['parameters'][zi]['name'] = nib
                #*ASSUMPTION:* all new dependencies should share the same offset as elements in the chain

            new_dependencies = []
            for z in x['depends']:
                if "initial_h2d" in z:
                    z = z+"-task"+str(d)
                    new_dependencies.append(z)
                    continue
                if "d2h" in z or "h2d" in z:
                    import ipdb
                    ipdb.set_trace()
                dependency_no = original_chain_length*d+int(z.replace('task',''))
                new_dependencies.append("task"+str(dependency_no))
            x['depends'] = new_dependencies
            y += 1 #increment the new task name counter
            #what about other (non-maximal) concurrency kernels?
            new_dag.append(x)

    #and update edges accordingly
    for d in range(0, _duplicates):#the maximum degree of concurrency indicates how many "parallel" DAGs to paste in
        for entry in edges:
            new_edges.append((str(int(entry[0])+original_chain_length*d),str(int(entry[1])+original_chain_length*d)))

    return new_dag,new_edges

def plot_dag(task_dag,edges,dag_path_plot):
    #edge_d = [(e[0],e[1],{'data':neighs_down_top[i]}) for i,e in enumerate(edges)]
    edge_d = [(e[0],e[1],{"depends":task_dag[int(e[0])]['name']}) for i, e in enumerate(edges)]

    #edge_d = [(e[i][], e['name'],{'data':e['name']}) for i,e in enumerate(task_dag)]
    dag = nx.DiGraph()
    #hash the colours in the dag for each kernel name and memory transfers
    unique_kernels = []
    for t in task_dag:
        for c in t['commands'][0]:
            if 'kernel' in c:
                unique_kernels.append(t['commands'][0][c]['name'])
            else:
                unique_kernels.append(t['name'])
    unique_kernels = set(unique_kernels)

    #first and last memory transfers have reserved blue (initial_h2d) and red (final_d2h) colours
    if 'initial_h2d' in unique_kernels: unique_kernels.remove('initial_h2d')
    if 'final_d2h' in unique_kernels: unique_kernels.remove('final_d2h')
    assert len(unique_kernels) < 7, "Can't colourize that many unique kernels -- maybe choose a bigger colour palette? <https://docs.bokeh.org/en/latest/docs/reference/palettes.html>"
    palette = brewer['Pastel1'][9]
    phash = {}
    phash['final_d2h']   = palette[0]
    phash['initial_h2d'] = palette[1]
    for i,k in enumerate(unique_kernels):
        phash[k] = palette[i+2]

    #unique shape logic
    unique_task_types = []
    for t in task_dag:
        if 'kernel' in t['commands'][0]:
            t['task_name'] = t['commands'][0]['kernel']['name']
            t['task_type'] = t['task_name'] #"Kernel"
        else:
            t['task_name'] = t['name']
            t['task_type'] = "memory transfer"
        unique_task_types.append(t['task_type'])

    unique_task_types = set(unique_task_types)
    shapes = []
    avail_shapes = {0:'s',1:'o',2:'*',3:'^',4:'v',5:'<',6:'>',7:'h',8:'H',9:'D',10:'d',11:'P',12:"X",13:'1',14:'p',15:'.'}
    assert len(unique_task_types) <= len(avail_shapes.items())
    for d in range(0,len(unique_task_types)):
        shapes.append(avail_shapes[d])
    #associate each kernel name with a shape
    node_shape_lookup = {}
    for i,d in enumerate(unique_task_types):
        node_shape_lookup[d] = shapes[i]
    #if there are edges, create the DAG from it, otherwise just the nodes (there are no edges in the DAG, and so the draw call will fail)
    #node_d = [(str(i),{"label":e['name'], "position":(i,0), "marker":node_shape_lookup[e['task_type']]}) for i, e in enumerate(task_dag)]
    node_d = [(str(i),{"label":e['name'], "position":(i,0)}) for i, e in enumerate(task_dag)]
    dag.add_nodes_from(node_d)
    node_labels = {str(i):'{}'.format(n['name']) for i,n in enumerate(task_dag)}
    #node_labels = {str(i):'' for i,n in enumerate(task_dag)}
    if edge_d != []:
        dag.add_edges_from(edge_d)
        edge_labels = {(e1,e2):'{}'.format(d) for e1,e2,d in dag.edges(data=True)}

    #plot it
    pos = graphviz_layout(dag,prog='dot')
    #add colour

    node_shapes = []
    node_colours = []
    for n in dag:
        if 'kernel' in task_dag[int(n)]['commands'][0]:
            node_colours.append('{}'.format(phash[task_dag[int(n)]['commands'][0]['kernel']['name']]))
        else: #h2d or d2h transfer tasks
            node_colours.append('{}'.format(phash[task_dag[int(n)]['name']]))
        node_shapes.append('{}'.format(node_shape_lookup[task_dag[int(n)]['task_type']]))

    fig = plt.figure(figsize=(3,9))
    ax = fig.add_subplot(111)
    nx.draw(dag,pos=pos,labels=node_labels,font_size=8,node_color=node_colours, node_shape=node_shapes)
    #`legend_handles = []
    #`for i,d in enumerate(unique_kernels):
    #`    import ipdb
    #`    ipdb.set_trace()
    #`    legend_handles.append(ax.scatter([], [],color='white', edgecolor='black', marker=node_shape_lookup[d], label=d))
    #`kernel_legend = ax.legend(handles=legend_handles,loc=1,title="Kernels",fontsize=8)
    #`if True:#show_kernel_legend:
    #`    plt.gca().add_artist(kernel_legend)
    #`    plt.tight_layout()
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
        for d in range(0,_duplicates):
            for ck in range(1,_concurrent_kernels[k]+1):
                instance = ck+_total_num_concurrent_kernels*d
                for i,j in enumerate(_kernel_buffs[k]):
                    if type(j) is dict and j['type'] == 'scalar' and type(j['value']) is str:
                        print("Error: we still need to add kernel argument passing to the runner, currently we only accept numerical values to the dagger_generator.py")
                        import ipdb
                        ipdb.set_trace()
                        import sys
                        sys.exit("Error: we still need to add kernel argument passing to the runner, currently we only accept numerical values to the dagger_generator.py")

                    if type(j) is dict and j['type'] == 'scalar':
                        continue
                    #inputs.append("hostmem-{}-buffer{}".format(k,i))
                    #TODO could facilitate concurrent-kernels with :
                    inputs.append("hostmem-{}-buffer{}-instance{}".format(k,i,instance))
    for k in _kernels:
        for d in range(0,_duplicates):
            for ck in range(1,_concurrent_kernels[k]+1):
                instance = ck+_total_num_concurrent_kernels*d
                for i,j in enumerate(_kernel_buffs[k]):
                    inputs.append("devicemem-{}-buffer{}-instance{}".format(k,i,instance))

    #TODO: support multiple targets--hint all but the h2d and d2h memory transfers should be transfer target (control-dependency) while all actual kernels should use the data-dependency
    inputs.append("user-target-control")
    inputs.append("user-target-data")
    #was "inputs":[ "user-size", "user-size-cb", "user-A", "user-B", "user-mem", "user-target" ]
    return inputs

def determine_and_prepend_iris_h2d_transfers(dag):
    if _use_data_memory:
        return dag
    #sample:       {
    #      "name" : "transferto0",
    #      "h2d": ["user-memA0", "user-A", "0", "user-size-cb"],
    #      "h2d": ["user-memB0", "user-B", "0", "user-size-cb"],
    #      "target": "user-target1"
    #  },
    transfers = []
    #this should be extended for each instance
    for i,k in enumerate(_kernels):
        for ck in range(1,_concurrent_kernels[k]+1):
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
                      if buffer_name == p['value']:
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

    if _use_data_memory:
        #iris_task_dmem_flush_out(task0,mem_C);
        #either we need to perform a data memory flush back on all of our short-listed pool of memory (since we can't be sure which objects have been written
        if _memory_object_pool is not None:
            #depend on all tasks before we flush
            all_tasks = []
            for t in dag:
                all_tasks.append(t['name'])

            #add a data-memory flush for each used memory object
            for buffer_name in _memory_object_pool:
                if _no_flush: continue
                commands = {}
                commands["d2h"] = {"name":"dmemflush-{}".format(buffer_name),"device_memory":buffer_name,"data-memory-flush":1}
                transfer = {}
                transfer["name"] = "dmemflush-{}".format(buffer_name)
                transfer["commands"] = [commands]
                transfer["depends"] = all_tasks
                transfer["target"] = "user-target-control"
                transfers.append(transfer)
        #or we need to flush just the written buffers
        else:
            for i,k in enumerate(_kernels):
                for ck in range(1,_concurrent_kernels[k]+1):
                    for j,l in enumerate(_kernel_buffs[k]):
                            if l == 'r':
                                continue
                            if _no_flush: continue

                            buffer_name = "devicemem-{}-buffer{}-instance{}".format(k,j,ck)
                            commands = {}
                            commands["d2h"] = {"name":"d2h-buffer{}-instance{}".format(j,ck),"device_memory":buffer_name,"data-memory-flush":1}
                            transfer = {}
                            transfer["name"] = "dmemflush-{}-buffer{}-instance{}".format(k,j,ck)
                            transfer["commands"] = [commands]
                            transfer["depends"] = []
                            #add this dependency to the DAGs--depend on all kernels which use these buffers
                            memory_instance_in_use = False
                            for m,t in enumerate(dag):
                                if 'kernel' not in t['commands'][0]:
                                    continue
                                for p in t['commands'][0]['kernel']['parameters']:
                                    if buffer_name == p['value']:
                                        memory_instance_in_use = True
                                        transfer["depends"].append(t["name"])
                            transfer["target"] = "user-target-control"
                            if memory_instance_in_use == True:
                                transfers.append(transfer)
        #prepend the d2h transfers
        for t in range(0, len(transfers)):
            dag.append(transfers.pop(0))

        return dag

    #we aren't using data memory --- process the explicit d2h memory transfers
    #this should be extended for each instance
    for i,k in enumerate(_kernels):
        for ck in range(1,_concurrent_kernels[k]+1):
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
                      if buffer_name == p['value']:
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
    f = open(_graph, 'w')
    inputs = determine_iris_inputs()
    if not _sandwich:
        dag = determine_and_prepend_iris_h2d_transfers(dag)
        dag = determine_and_append_iris_d2h_transfers(dag)
    final_json = {"$schema": __schema__, "iris-graph":{"inputs":inputs,"graph":{"tasks":dag}}}
    f.write(json.dumps(final_json,indent = 2))
    f.close()

def get_task_graph_from_json(file_url):
    with open(file_url, 'r') as json_file:
        content = json.load(json_file)
    return content

def run():
    random.seed(_seed)
    task_per_level, level_per_task = gen_task_nodes(_depth,_num_tasks,_min_width,_max_width)
    edges,neighs_top_down,neighs_down_top = gen_task_links(_mean,_std_dev,task_per_level,level_per_task,delta_lvl=_skips)
    if _sandwich:
        neighs_down_top = repack_dag_with_missing_edges(neighs_down_top,neighs_top_down)
    task_dag = generate_attributes(neighs_down_top,task_per_level)
    task_dag = rename_special_tasks(task_dag)
    edges = prune_edges_from_dependencies(task_dag,edges)
    task_dag, edges = duplicate_for_concurrency(task_dag,edges)
    plot_dag(task_dag,edges,'./dag.pdf')
    get_task_to_json(task_dag,edges)

def main():
    run()


if __name__ == '__main__':
    parse_args()

    main()

    print('done')
