#!/usr/bin/env python3

import iris
import numpy as np
import sys

# IRIS initialize
iris.init()

N=16
# Create and initialize src and dst memory
src_data = np.arange(N, dtype=np.float32)
dst_data = np.zeros(N, dtype=np.float32)

# Create DMEM2DMEM command in task
src = iris.dmem(src_data)
dst = iris.dmem(dst_data)

# Create task
task = iris.task()

# Add DMEM2DMEM command to task
task.dmem2dmem(src, dst)

# Add flush command to task
task.flush(dst)

# Submit task
task.submit()

# Compare output
print(np.all(src_data == dst_data))

# IRIS finalize
iris.finalize()