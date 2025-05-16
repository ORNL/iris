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

qir_content = iris.dmem_infile("bell.ll")
#print("content: ", qir_content.host.tobytes().decode('latin-1'))
num_shots = 128
output = iris.dmem()
backend = iris.dmem("qsim")

# Running in parallel
n = 1

# Create n tasks (no communications with each other)
tasks = [iris.task() for i in range(n)]

# Feed parameters to each task and submit
# sync tells you how long to wait for completion of the command; 0=do not wait
# single task submission: task1.submit(iris.iris_cpu)
for i in range(n):
    #qiree(qir_content, num_shots, output, backend)
    tasks[i] = iris.task("qiree", 1, [], [1], [], [
#        (mem_i, iris.iris_r),
        (qir_content, iris.iris_r), num_shots,
        (output, iris.iris_w),
        (backend, iris.iris_r)
#        (mem_s, iris.iris_r),
#        (mem_s_num, iris.iris_r)
    ])
    tasks[i].submit(iris.iris_default, sync=0)
iris.synchronize()
iris.finalize()
