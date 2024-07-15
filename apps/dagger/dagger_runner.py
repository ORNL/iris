#!/usr/bin/env python3

"""
Adapted from dagger_runner.cpp
"""
__author__ = "Aaron Young, Beau Johnston"
__copyright__ = "Copyright (c) 2020-2023, Oak Ridge National Laboratory (ORNL) Programming Systems Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "1.0"

import iris
import argparse
import re
import numpy as np
import sys
from . import dagger_generator as dg
import pdb

EXIT_FAILURE = 1

scheduling_policy_lookup = {
    "roundrobin": iris.iris_roundrobin,
    "depend"    : iris.iris_depend,
    "data"      : iris.iris_data,
    "profile"   : iris.iris_profile,
    "random"    : iris.iris_random,
    "sdq"       : iris.iris_sdq,
    "ftf"       : iris.iris_ftf,
    "custom"    : iris.iris_custom,
}

def init_parser(parser):
    parser.add_argument("--size",required=True,type=int,help="The size of the memory buffers to use in this IRIS test.")
    parser.add_argument("--repeats",required=True,type=int,help="The number of repeats.")
    parser.add_argument("--logfile",required=True,type=str,help="The location to log the timing results.")
    parser.add_argument("--scheduling-policy",default='roundrobin',help="all options include (roundrobin, depend, profile, random, sdq, ftf, custom) or any integer [0-9] denoting the device id to run all tasks on.")
    parser.add_argument("--attach-debugger",action='store_true',help="Attach debugger on port 5678.")

def create_graph(args):

    num_buffers_used = 0
    host_mem = []
    buffer_type = []
    dev_mem = []
    memory_task_target = iris.iris_pending
    sizecb = []
    input_arrays = {
        "host_mem": host_mem,
        "buffer_type": buffer_type,
        "dev_mem": dev_mem,
        "sizecb": sizecb,
    }

    for kernel in dg._kernels:
        #TODO: update this in line to the use of concurrent_kernels and duplicates
        for concurrent_device in range(dg._concurrent_kernels[kernel]):
            argument_index = 0
            for buffer in dg._kernel_buffs[kernel]:
                # Create and add the host-side buffer based on it's type
                if buffer == 'r':
                    # Create and populate memory
                    tmp = np.arange(args.size**dg._dimensionality[kernel], dtype=np.double)
                    for i in range(args.size**dg._dimensionality[kernel]):
                        tmp[i] = i
                    host_mem.append(tmp)
                    num_buffers_used += 1
                elif buffer == 'w':
                    tmp = np.arange(args.size**dg._dimensionality[kernel], dtype=np.double)
                    host_mem.append(tmp)
                    num_buffers_used += 1
                elif buffer == 'rw':
                    # Create and populate memory
                    tmp = np.arange(args.size**dg._dimensionality[kernel], dtype=np.double)
                    for i in range(args.size**dg._dimensionality[kernel]):
                        tmp[i] = i
                    host_mem.append(tmp)
                    num_buffers_used += 1
                else:
                    print(f"\033[41mInvalid memory argument! Kernel {kernel} has a buffer of memory type {buffer} but only r,w or rw are allowed.\n\033[0m")
                    exit(EXIT_FAILURE)
                buffer_type.append(buffer)
                if args.use_data_memory:
                    iris_mem = iris.dmem(host_mem[-1])
                    memory_task_target = iris.iris_default
                else:
                    iris_mem = iris.mem(host_mem[-1].nbytes)
                dev_mem.append(iris_mem)

        sizecb.append(args.size**dg._dimensionality[kernel]*np.double(0).itemsize)
        print(f"SIZE[{args.size}] MATRIX_SIZE[{sizecb[-1]/1024/1024}]MB")

    json_inputs = []

    json_inputs.append(args.size)
    json_inputs.extend(sizecb)
    json_inputs.extend(host_mem)
    json_inputs.extend(dev_mem)
    json_inputs.append(memory_task_target)
    json_inputs.append(args.task_target)

    print("JSON input parameters")
    for index, inp in enumerate(json_inputs):
        print(index,':', inp)

    graph = iris.graph()
    graph.load(args.graph, json_inputs)
    return graph, input_arrays

def run(args):
    print(f"REPEATS:{args.repeats} LOGFILE:{args.logfile} POLICY:{args.scheduling_policy}")

    for k in dg._kernels:
        print(f"KERNEL: {k} available on {dg._concurrent_kernels[k]} devices concurrently")

    iris.init()
    for t in range(args.repeats):
        graph, input_arrays = create_graph(args)
        graph.submit()
        graph.wait()
    print("Success")
    iris.finalize()

def parse_args(pargs=None):
    # Parse the arguments
    args = dg.parse_args(pargs=pargs, additional_arguments=[init_parser])
    args.task_target = scheduling_policy_lookup[args.scheduling_policy]
    return args

def main():

    args = parse_args() 
    if args.attach_debugger:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        # debugpy.breakpoint()
    run(args)

if __name__ == '__main__':
    main()
