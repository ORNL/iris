#!/usr/bin/env python3

import iris
import pdb
import os

# Step1: MLIR circuit cutting pass (Richard, Sharmin, Daniel, Elaine, Narasinga)
## will generate one MLIR file each fragment 
## Generate Metadata.cpp and Metadata.txt files for dependencies and validity functions 
## Optional: Generate Post-processing function post_processing.cpp
# Step2: Pennylane tool to convert MLIR file to QIR file (Sharmin, Daniel)
# Step3: Post-processing function manually written (Richard, Vicente)


#Metadata.cpp: To provide validity functions for task invokation
#bool cond1(<state-of-circuit>)
#{
#    return true;
#}

#Metadata.txt : to represent dependencies
# fragment5: fragment1[<cond1>], fragment2[<cond1>]
# 

os.environ["IRIS_ARCHS"] = "openmp:qiree"
os.system("make")
iris.init()

# This is the home folder where the built QIR-EE repo lies
home = os.getenv("QIREE")
if not (home and os.path.exists(home)):
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
