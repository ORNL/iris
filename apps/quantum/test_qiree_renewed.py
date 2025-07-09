#!/usr/bin/env python3

import iris
import pdb
import os

os.environ["IRIS_ARCHS"] = "openmp:qiree"

iris.init()

# This is the home folder where the built QIR-EE repo lies
home = os.getenv("QIREE")
if not os.path.exists(home):
    home = os.path.join(os.getenv("HOME"), "/qiree")
# Parameters should be a list of strings: [executable, LL file, name of accelerator to use, number of shots]

#print("content: ", qir_content.host.tobytes().decode('latin-1'))
num_shots = 128
backend = iris.dmem("qsim")

num_cuts = 2
# Running in parallel
n = num_cuts

# Create n tasks (no communications with each other)
tasks = [iris.task() for i in range(n)]
output = [iris.dmem() for i in range(n)]
qir_content = [ iris.dmem_infile(f"bell{i}.ll") for i in range(n) ]


# Feed parameters to each task and submit
# sync tells you how long to wait for completion of the command; 0=do not wait
# single task submission: task1.submit(iris.iris_cpu)
for i in range(n):
    tasks[i] = iris.task("qiree", 1, [], [1], [], [
        (qir_content[i], iris.iris_r), num_shots,
        (output[i], iris.iris_w),
        (backend, iris.iris_r)
    ])
    tasks[i].submit(iris.iris_qpu, sync=0)

# Classical compute task to do post-processing
result = iris.dmem()
post_process_task = iris.task("qiree_post_processing", 1, [], [1], [], 
        [(output[i], iris.iris_r) for i in range(n)] + [ (result, iris.iris_w) ]
)
post_process_task.depends(tasks)
post_process_task.submit(iris.iris_cpu, sync=0)

iris.synchronize()
print("Result:", result)

iris.finalize()
