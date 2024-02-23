#!/usr/bin/env python3

import iris
import numpy as np
import sys
import pdb

iris.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10
SIZEK=np.int32(8)

x = np.arange(SIZE, dtype=np.int32)
y = np.arange(SIZE, dtype=np.int32)
s = np.arange(SIZE, dtype=np.int32)

print('X', x)
print('Y', y)
print('A', A)
print('SIZE', SIZE)

mem_x = iris.mem(x.nbytes)
mem_y = iris.mem(y.nbytes)
mem_s = iris.mem(s.nbytes)

task = iris.task()
task.h2d(mem_x, 0, x.nbytes, x)
task.h2d(mem_y, 0, y.nbytes, y)
#pdb.set_trace()
task.kernel("saxpy", 1, [0], [SIZE], [1], 
        [mem_s,       mem_x,       mem_y,       SIZE,  A] , 
        [iris.iris_w, iris.iris_r, iris.iris_r, 4,     4] )
task.params_map([iris.iris_ftf, iris.iris_ftf, iris.iris_ftf, iris.iris_cpu, iris.iris_ftf])
task.d2h(mem_s, 0, s.nbytes, s)
task.submit(iris.iris_fpga)

print('S =', A, '* X + Y', s)

iris.finalize()

