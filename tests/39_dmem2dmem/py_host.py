#!/usr/bin/env python3

import iris
import numpy as np
import sys

iris.init()

N=16
src = np.arange(N, dtype=np.float32)
dst = np.zeros(N, dtype=np.float32)


src_iris = iris.dmem(src)
dst_iris = iris.dmem(dst)

task = iris.task()
task.dmem2dmem(src_iris, dst_iris)
task.flush(dst_iris)
task.submit()

print(np.all(src == dst))