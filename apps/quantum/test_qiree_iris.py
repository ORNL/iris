#!/usr/bin/env python3

import iris
import pdb
import os

iris.init()

def go_quantum(iris_params, iris_dev):

    params = iris.iris2py(iris_params)
    dev = iris.iris2py_dev(iris_dev)
    # print(params)
    
    # QIR-EE Inputs
    execpath = params[0]
    filepath = params[1]
    accelerator = params[2]
    numshots = params[3]

    # QIR-EE Execution
    results = os.system(execpath+" "+filepath+" -a "+accelerator+" -s "+numshots+" > output.txt")
    
    # Reading and printing measurement results
    lines = []
    with open('output.txt') as f:
        [lines.append(line) for line in f.readlines()]
        # Extract the lines with measurement results
        # This only works for bell.ll (we can generalize later)
        clean = [i.replace(" ", "") for i in lines[6:8]]
        clean = [i.replace("\n", "") for i in clean]
        print(clean[0]+" "+clean[1])
    
    return iris.iris_ok

# This is the home folder where the built QIR-EE repo lies
home = os.getenv("QIREE")
if not os.path.exists(home):
    home = os.path.join(os.getenv("HOME"), "/qiree")
# Parameters should be a list of strings: [executable, LL file, name of accelerator to use, number of shots]
parameters = [home+"/build/bin/qir-xacc", home+"/examples/bell.ll", "qpp", "1024"]

mem_ll = iris.dmem("bell.ll")
mem_a = iris.dmem("-a")
mem_qpp = iris.dmem("qpp")


# Running in parallel
n = 64

# Create n tasks (no communications with each other)
tasks = [iris.task() for i in range(n)]

# Feed parameters to each task and submit
# sync tells you how long to wait for completion of the command; 0=do not wait
# single task submission: task1.submit(iris.iris_cpu)
for i in range(n):
    tasks[i] = iris.task("qiree", 1, [], [1], [], [
        (mem_ll, iris.iris_r),
        (mem_a, iris.iris_r),
        (mem_qpp, iris.iris_r)
    ])
    tasks[i].submit(iris.iris_default, sync=0)
iris.synchronize()
iris.finalize()
