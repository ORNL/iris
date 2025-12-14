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


# ---------- Helper functions for AcceleratorBuffer processing ----------

def save_accelerator_buffer_to_file(qpu_buffer: Any, 
                                     filepath: str, 
                                     fragment_idx: int, 
                                     circuit_idx: int) -> None:
    """
    Save the XACC AcceleratorBuffer dictionary output to a file for future parsing.
    
    Parameters
    ----------
    qpu_buffer : Any
        The structured array containing the QPU result with 'out' field.
    filepath : str
        Path to the file where results will be appended (JSON Lines format).
    fragment_idx : int
        Index of the circuit fragment.
    circuit_idx : int
        Index of the subcircuit within the fragment.
    """
    try:
        # Extract the dictionary from the 'out' field if it exists
        accelerator_dict = None
        
        if hasattr(qpu_buffer, 'dtype') and hasattr(qpu_buffer.dtype, 'names'):
            if qpu_buffer.dtype.names and 'out' in qpu_buffer.dtype.names:
                if len(qpu_buffer) > 0:
                    out_json_str = str(qpu_buffer['out'][0])
                    try:
                        accelerator_dict = json.loads(out_json_str)
                    except json.JSONDecodeError:
                        pass
        
        if accelerator_dict:
            # Append to file in JSON Lines format with metadata
            record = {
                "fragment_idx": fragment_idx,
                "circuit_idx": circuit_idx,
                "buffer": accelerator_dict
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'a') as f:
                f.write(json.dumps(record) + '\n')
        else:
            print(f"  [Fragment {fragment_idx}, Circuit {circuit_idx}] Warning: Could not extract AcceleratorBuffer")
    
    except Exception as e:
        print(f"  [Fragment {fragment_idx}, Circuit {circuit_idx}] Error saving buffer: {e}")


def extract_measurements_from_dict(qpu_buffer_dict: dict) -> Any:
    """
    Extract the "Measurements" key from an XACC AcceleratorBuffer dictionary.
    
    Parameters
    ----------
    qpu_buffer_dict : dict
        Dictionary containing the AcceleratorBuffer data, typically with structure:
        {"AcceleratorBuffer": {"Measurements": {...}, ...}}
    
    Returns
    -------
    measurements : Any
        The extracted Measurements data, or the original dict if extraction fails.
    """
    try:
        if isinstance(qpu_buffer_dict, dict):
            if 'AcceleratorBuffer' in qpu_buffer_dict:
                if 'Measurements' in qpu_buffer_dict['AcceleratorBuffer']:
                    return qpu_buffer_dict['AcceleratorBuffer']['Measurements']
    except (KeyError, TypeError):
        pass
    
    # Fallback: return the original dict if extraction fails
    return qpu_buffer_dict


def extract_measurements_from_iris_output(iris_stdout: str, num_fragments: int, num_subcircuits: int = 3) -> list:
    """
    Parse IRIS stdout output and extract Measurements data from printed AcceleratorBuffer dictionaries.
    Handles multi-line JSON blocks prefixed by 'out:' as observed in IRIS output.

    Parameters
    ----------
    iris_stdout : str
        Captured stdout from iris.synchronize() containing printed AcceleratorBuffer dictionaries.
    num_fragments : int
        Number of circuit fragments expected.
    num_subcircuits : int
        Number of subcircuits per fragment (default 3).

    Returns
    -------
    all_measurements : list
        List of extracted Measurements from all fragments and subcircuits.
    """
    all_measurements = []

    lines = iris_stdout.split('\n')

    # State machine to capture 'out:{ ... }' JSON blocks
    capturing = False
    brace_depth = 0
    current_block_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()

        # Detect start of JSON block preceded by 'out:'
        if not capturing:
            if 'out:' in line:
                # Everything after 'out:' begins a JSON object
                prefix_idx = line.find('out:')
                after = line[prefix_idx + len('out:'):].lstrip()
                if after:
                    # If same-line JSON start
                    current_block_lines = [after]
                    # Initialize brace depth based on this line
                    brace_depth = after.count('{') - after.count('}')
                else:
                    # Next lines will contain JSON
                    current_block_lines = []
                    brace_depth = 0
                capturing = True
                continue
            else:
                # Also allow blocks that start directly with a JSON object (rare)
                if line.strip().startswith('{'):
                    current_block_lines = [line.strip()]
                    brace_depth = line.count('{') - line.count('}')
                    capturing = True
                    continue
                # Not a JSON start line; skip
                continue

        # While capturing, accumulate lines and track braces
        current_block_lines.append(line)
        brace_depth += line.count('{') - line.count('}')

        # End of JSON block when braces balanced
        if brace_depth <= 0:
            capturing = False
            json_str = '\n'.join(current_block_lines).strip()
            # Attempt to parse the block as JSON
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and 'AcceleratorBuffer' in data:
                    measurements = extract_measurements_from_dict(data)
                    all_measurements.append(measurements)
                    print(f"  ✓ Extracted Measurements (count: {len(all_measurements)})")
            except json.JSONDecodeError:
                # Try Python literal dict (single quotes)
                try:
                    data = ast.literal_eval(json_str)
                    if isinstance(data, dict):
                        all_measurements.append(data)
                        print(f"  ✓ Extracted Measurements (literal) (count: {len(all_measurements)})")
                except Exception:
                    # Failed to parse; ignore this block
                    pass
            # Reset for next block
            current_block_lines = []
            brace_depth = 0

    return all_measurements

# Set up QIREE library path and IRIS devices
home_dir = os.path.expanduser("~")
os.environ["IRIS_LIB_QIREE"] = os.path.join(home_dir, "qiree/build/lib/libcqiree.so")
os.environ["LIB_QIREE"] = os.environ["IRIS_LIB_QIREE"]
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
num_fragments_test = 2
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
tasks = [[iris.task() for j in range(3)] for i in range(num_fragments)]
output = [[iris.dmem(np.empty(0, dtype=iris.CQireeResultRecord)) for j in range(3)] for i in range(num_fragments)]
# Load QIR file paths
home_dir = os.path.expanduser("~")
qir_paths = [[os.path.join(home_dir, f"qcut/qcut_circuits_qir_only/circ_{i:02}_{j}_qir.ll")
              for j in range(3)] for i in range(num_fragments)]
qir_content = [[iris.dmem_infile(qir_paths[i][j])
                for j in range(3)] for i in range(num_fragments)]

num_fragments_test = 2

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
        tasks[i][j] = iris.task("qiree", 1, [], [1], [], [
            (backend_handle, iris.iris_r),
            (qir_content[i][j], iris.iris_r),  # Correct subcircuit
            (config_handle, iris.iris_r),
            backends_config[i]["shots"],
            (output[i][j], iris.iris_w),  # Correct output buffer
        ])
        # Async mode for parallel execution
        tasks[i][j].submit(iris.iris_qpu, sync=0)
        if backends_config[i]["backend"] != "xacc":
            print(f"    Submitted ({backends_config[i]['backend']})")
        else:
            print(f"    Submitted ({backends_config[i]['backend']}/{backends_config[i]['accelerator']})")

# Classical compute: Extract all circuit results and estimate ⟨ZZZZ⟩
print(f"\nWaiting for circuit execution to complete...")

# Output directory for saving AcceleratorBuffer dictionaries
output_dir = "./qcut_accelerator_buffers"
Path(output_dir).mkdir(parents=True, exist_ok=True)
iris_output_file = os.path.join(output_dir, "iris_stdout.txt")

# Redirect file descriptors (1=stdout, 2=stderr) to capture C-level output
print("Capturing IRIS output to file (redirecting file descriptors)...")
sys.stdout.flush()
sys.stderr.flush()

# Open the output file and save the original file descriptors
with open(iris_output_file, 'w') as iris_file:
    # Save original stdout and stderr file descriptors
    saved_stdout_fd = os.dup(1)  # Duplicate stdout (fd 1)
    saved_stderr_fd = os.dup(2)  # Duplicate stderr (fd 2)
    
    try:
        # Redirect both stdout and stderr to the file
        os.dup2(iris_file.fileno(), 1)  # Redirect fd 1 (stdout) to file
        os.dup2(iris_file.fileno(), 2)  # Redirect fd 2 (stderr) to file
        
        # Wait for all circuit execution tasks to complete
        # This output will now go to the file
        iris.synchronize()
        
    finally:
        # Flush and close the file
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Restore original stdout and stderr
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

# Restore Python's stdout/stderr objects
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Read back the saved output
with open(iris_output_file, 'r') as f:
    iris_output = f.read()

print(f"✓ Saved IRIS stdout/stderr to {iris_output_file}")

print(f"\nExtracting circuit results from IRIS stdout...")

# Extract Measurements from the saved IRIS output
all_circuit_results = extract_measurements_from_iris_output(iris_output, num_fragments_test, num_subcircuits=3)

print(f"✓ Extracted {len(all_circuit_results)} results from {num_fragments_test} fragments ({num_fragments_test * 3} subcircuits)")

# Persist extracted circuit results to a separate file for inspection
circuit_results_file = os.path.join(output_dir, "circuit_results.jsonl")
try:
    with open(circuit_results_file, 'w') as f:
        # Save as JSON Lines: one measurement per line
        for idx, meas in enumerate(all_circuit_results):
            # Ensure JSON-serializable
            try:
                record = {
                    "index": idx,
                    "fragment": idx // 3,
                    "subcircuit": idx % 3,
                    "measurement": meas,
                }
                f.write(json.dumps(record) + "\n")
            except TypeError:
                # Fallback: convert numpy arrays or other types
                record = {
                    "index": idx,
                    "fragment": idx // 3,
                    "subcircuit": idx % 3,
                    "measurement": np.array(meas).tolist() if hasattr(meas, "__array__") else str(meas),
                }
                f.write(json.dumps(record) + "\n")
    print(f"✓ Saved extracted circuit results to {circuit_results_file}")
except Exception as e:
    print(f"✗ Failed to save circuit results: {e}")

# Sanity check: ensure we captured 3 subcircuits per fragment
expected = num_fragments_test * 3
if len(all_circuit_results) != expected:
    missing = expected - len(all_circuit_results)
    print(f"Parsed {len(all_circuit_results)} results but expected {expected}. Missing {missing}.")
    print("   Note: IRIS may print multiple AcceleratorBuffers per subcircuit; parser captures each complete 'out: { ... }' block in order.")
else:
    print("✓ Verified: 3 subcircuits per fragment captured for all fragments.")

iris.finalize()
