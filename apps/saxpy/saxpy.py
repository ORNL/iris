#!/usr/bin/env python3

import iris
import numpy as np
import sys

iris.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
s = np.arange(SIZE, dtype=np.float32)

print('X', x)
print('Y', y)

mem_x = iris.mem(x.nbytes)
mem_y = iris.mem(y.nbytes)
mem_s = iris.mem(s.nbytes)

task = iris.task()
task.h2d_full(mem_x, x)
task.h2d_full(mem_y, y)
task.kernel("saxpy", 1, [], [SIZE], [], [mem_s, A, mem_x, mem_y] , [iris.iris_w, 4, iris.iris_r, iris.iris_r] )
task.d2h_full(mem_s, s)
task.submit(iris.iris_gpu)

print('S =', A, '* X + Y', s)

iris.finalize()

