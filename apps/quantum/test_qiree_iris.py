#!/usr/bin/env python3

import iris
import pdb
import os

iris.init()

# This is the home folder where the built QIR-EE repo lies
home = os.getenv("QIREE")
if not os.path.exists(home):
    home = os.path.join(os.getenv("HOME"), "/qiree")
# Parameters should be a list of strings: [executable, LL file, name of accelerator to use, number of shots]

mem_ll = iris.dmem("bell.ll")
mem_a = iris.dmem("-a")
mem_qpp = iris.dmem("qpp")

#mem_i = iris.dmem("-i")
#mem_ll = iris.dmem("bell.ll")
#mem_s = iris.dmem("-s")
#mem_s_num = iris.dmem("1024")


# Running in parallel
n = 64

# Create n tasks (no communications with each other)
tasks = [iris.task() for i in range(n)]

# Feed parameters to each task and submit
# sync tells you how long to wait for completion of the command; 0=do not wait
# single task submission: task1.submit(iris.iris_cpu)
for i in range(n):
    tasks[i] = iris.task("qiree", 1, [], [1], [], [
#        (mem_i, iris.iris_r),
        (mem_ll, iris.iris_r),
        (mem_a, iris.iris_r),
        (mem_qpp, iris.iris_r)
#        (mem_s, iris.iris_r),
#        (mem_s_num, iris.iris_r)
    ])
    tasks[i].submit(iris.iris_default, sync=0)
iris.synchronize()
iris.finalize()
