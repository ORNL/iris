#!/usr/bin/env python3
"""
Processing the GHZ QCUT example using Q-IRIS and QIR files
"""

import iris
import os
import time
import numpy as np
import json
from post_processing_ghz import *

# Set up QIREE library path and IRIS devices
os.environ["LIB_QIREE"] = "/home/kew/qiree/build/lib/libcqiree.so"
os.environ["IRIS_ARCHS"] = "openmp:qiree"
os.environ["IRIS_QIREE_DEVICE_FACTOR"] = "1"
os.environ["IRIS_PROFILE"] = "1"

# Initialize IRIS
iris.init()

# Configuration
# Shots configurable via environment: export NUM_SHOTS=100
num_shots = int(os.getenv("NUM_SHOTS", "100"))

# Backend configuration for each circuit fragment
# We can specify different backends/accelerators for each fragment 
# but since we have a lot of circuits we only demo 1 for functionality.

num_fragments = 4

# This sets up a config for all fragments to use the same backend.
backends_config = [
    {"backend": "xacc", "accelerator": "qsim", "shots": num_shots}
 for i in range(num_fragments)]

print("=" * 70)
print("GHZ QCut Example")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Backends Used: {num_fragments}")
print(f"  Shots per circuit: {num_shots}")
print(f"  Backend Config: backend={backends_config[0]['backend']}, accelerator={backends_config[0]['accelerator']}, shots={backends_config[0]['shots']}")

# Prepare configuration as JSON string
print(f"\nPreparing backend configurations...")

# Create IRIS objects for each backend
backends = [iris.dmem(cfg["backend"]) for cfg in backends_config]

config_jsons = []
for cfg in backends_config:
    if cfg["backend"] != "xacc":
        config_jsons.append(iris.dmem("null"))
    else:
        config_jsons.append(iris.dmem(json.dumps({"accelerator": cfg["accelerator"], "shots": cfg["shots"]})))

print(f"\nRunning {num_fragments} circuit fragments...")
print("-" * 70)

# Create n tasks (no communications with each other)
tasks = [iris.task() for i in range(num_fragments)]
output = [[iris.dmem(np.empty(0, dtype=iris.CQireeResultRecord)) for j in range (3)] for i in range(num_fragments)]
qir_content = [[iris.dmem_infile(f"/home/kew/qcut/qcut_circuits_qir_only/circ_{i:02}_{j}_qir.ll") 
                       for j in range (3)] for i in range(num_fragments)]

print(f"✓ Created {num_fragments} IRIS tasks")
print(f"✓ Loaded {num_fragments*3} QIR circuits")

# Feed parameters to each task and submit
for i in range(num_fragments):
    for j in range(3):
        tasks[i] = iris.task("qiree", 1, [], [1], [], [
            (backends[i], iris.iris_r),
            (qir_content[i][j], iris.iris_r),
            (config_jsons[i], iris.iris_r),
            backends_config[i]["shots"],
            (output[i][j], iris.iris_w),
            ]
        )       
    tasks[i].submit(iris.iris_qpu, sync=0)
    if backends_config[i]["backend"] != "xacc":
        print(f"  [Fragment {i}] Submitted to {backends_config[i]['backend']} backend")
    else:
        print(f"  [Fragment {i}] Submitted to {backends_config[i]['backend']} ({backends_config[i]['accelerator']}) backend")

# Classical compute task to do post-processing
# Create result buffer for output
result_buffer = iris.dmem("")

# Build list of all circuit outputs from all fragments
# Each fragment contributes 3 circuits
circuit_outputs = []
for i in range(num_fragments):
    for j in range(3):
        circuit_outputs.append((output[i][j], iris.iris_r))

# Add result buffer for output
circuit_outputs.append((result_buffer, iris.iris_w))

# Pass all circuit outputs to the post-processing function
post_process_task = iris.task(post_processing, 1, [], [1], [], circuit_outputs)
post_process_task.depends(tasks)
post_process_task.submit(iris.iris_cpu, sync=0)
print(f"✓ Submitted post-processing task")

# Wait for completion
print(f"\nWaiting for execution...")
iris.synchronize()

print(f"\n✓ All tasks completed successfully!")
iris.finalize()
