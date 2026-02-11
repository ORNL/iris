#!/usr/bin/env python3
"""
Processing the GHZ QCUT example using Q-IRIS and QIR files
"""

import iris
import os
import time
import sys
import io
import numpy as np
import json
from pathlib import Path
from typing import Any
import ast
import pdb


# Set up QIREE library path and IRIS devices
home_dir = os.path.expanduser("~")
os.environ["IRIS_LIB_QIREE"] = os.path.join(home_dir, "qiree/build/lib/libcqiree.so")
os.environ["IRIS_ARCHS"] = "openmp:qiree"
os.environ["IRIS_QIREE_DEVICE_FACTOR"] = "8"
os.environ["IRIS_PROFILE"] = "1"

# Suppress verbose XACC output
os.environ["XACC_VERBOSE"] = "0"

# Initialize IRIS
iris.init()

# Configuration
# Shots configurable via environment: export NUM_SHOTS=100
num_shots = int(os.getenv("NUM_SHOTS", "100"))
num_fragments_test = 12
num_fragments = num_fragments_test

# This sets up a config for all fragments to use the same backend.
backends_config = [
    {"backend": "xacc", "accelerator": "qpp", "shots": num_shots}
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

# Note: Create backend/config handles per submission to avoid reusing the same dmem,
# which can cause repeated execution of the first circuit when handles are shared.
def make_backend_handle(cfg):
    return iris.dmem(cfg["backend"]) if cfg["backend"] != "" else iris.dmem("xacc")

def make_config_handle(cfg):
    if cfg["backend"] != "xacc":
        return iris.dmem("null")
    return iris.dmem(json.dumps({"accelerator": cfg["accelerator"], "shots": cfg["shots"]}))

# Create one task per subcircuit (2D array: fragment x subcircuit)
tasks = np.array([[iris.task() for j in range(3)] for i in range(num_fragments)])
qpu_output = [[iris.dmem(np.empty(0, dtype=iris.CQireeResultRecord)) for j in range(3)] for i in range(num_fragments)]
output_w = [[(qpu_output[i][j], iris.iris_w) for j in range(3)] for i in range(num_fragments)]
inputs_r = [(qpu_output[i][j], iris.iris_r) for j in range(3) for i in range(num_fragments)]

# Load QIR file paths
home_dir = os.path.expanduser("~")
qir_paths = [[os.path.join(home_dir, f"qcut/qcut_circuits_qir_only/circ_{i:02}_{j}_qir.ll")
              for j in range(3)] for i in range(num_fragments)]
qir_content = [[iris.dmem_infile(qir_paths[i][j])
                for j in range(3)] for i in range(num_fragments)]

def post_processing(spec, *inputs_r):
    print(f"\nExtracting circuit results from IRIS ...")

    inputs_r_np = iris.CQireeResult.reshape(inputs_r, [num_fragments, 3])
    print("all outputs:", inputs_r_np.shape, " output:", inputs_r_np)
    
    # Create output directory if it doesn't exist
    os.makedirs("qcut_accelerator_buffers", exist_ok=True)
    
    # Write results to files
    with open("qcut_accelerator_buffers/iris_stdout.txt", "w") as f:
        f.write(f"Extracted circuit results from IRIS\n")
        f.write(f"Total outputs: {len(inputs_r)} results\n")
        f.write(f"Shape: {inputs_r_np.shape}\n")
        f.write(f"Fragments: {inputs_r_np.shape[0]}, Subcircuits per fragment: {inputs_r_np.shape[1]}\n\n")
        for fragment in range(inputs_r_np.shape[0]):
            for index in range(inputs_r_np.shape[1]):
                f.write(f"Fragment {fragment}, Subcircuit {index}:\n{inputs_r_np[fragment][index]}\n")
    
    with open("qcut_accelerator_buffers/circuit_results.jsonl", "w") as f:
        for fragment in range(inputs_r_np.shape[0]):
            for index in range(inputs_r_np.shape[1]):
                result_dict = {
                    "fragment": fragment,
                    "subcircuit": index,
                    "result": str(inputs_r_np[fragment][index])
                }
                f.write(json.dumps(result_dict) + "\n")
    
    print(f"✓ Extracted {len(inputs_r)} results from {inputs_r_np.shape[0]} fragments ({inputs_r_np.shape[0] * inputs_r_np.shape[1]} subcircuits)")
    print(f"✓ Results written to qcut_accelerator_buffers/iris_stdout.txt and circuit_results.jsonl")

print(f"✓ Created {num_fragments_test} tasks")
print(f"✓ Loaded {num_fragments_test*3} QIR circuits")

# Feed parameters: submit all 3 subcircuits per fragment
print(f"\nSubmitting {num_fragments_test * 3} subcircuits...")
for i in range(num_fragments_test):
    for j in range(3):
        # Debug: print which QIR file is being submitted
        try:
            from hashlib import sha256
            with open(qir_paths[i][j], 'rb') as fh:
                data = fh.read()
            digest = sha256(data).hexdigest()[:16]
            print(f"  [Fragment {i}, Subcircuit {j}] -> {qir_paths[i][j]} (sha256: {digest})")
        except Exception as e:
            print(f"  [Fragment {i}, Subcircuit {j}] Warning: Could not read QIR file: {e}")

        backend_handle = make_backend_handle(backends_config[i])
        config_handle = make_config_handle(backends_config[i])
        tasks[i][j].kernel("qiree", 1, [], [1], [], [
            (backend_handle, iris.iris_r),
            (qir_content[i][j], iris.iris_r),  # Correct subcircuit
            (config_handle, iris.iris_r),
            backends_config[i]["shots"],
            output_w[i][j],  # Correct output buffer
        ])
        # Async mode for parallel execution
        tasks[i][j].submit(iris.iris_qpu, sync=0)
        if backends_config[i]["backend"] != "xacc":
            print(f"    Submitted ({backends_config[i]['backend']})")
        else:
            print(f"    Submitted ({backends_config[i]['backend']}/{backends_config[i]['accelerator']})")

# Classical compute: Extract all circuit results and estimate ⟨ZZZZ⟩
print(f"\nWaiting for circuit execution to complete...")

post_process_task = iris.task(post_processing, 1, [], [1], [], inputs_r)
post_process_task.depends(tasks.reshape(-1))
post_process_task.submit(iris.iris_cpu, sync=1)

# Extract Measurements from the saved IRIS output


iris.finalize()
