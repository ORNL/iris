#!/usr/bin/env python3

import brisbane
import numpy as np
import sys

brisbane.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
s = np.arange(SIZE, dtype=np.float32)

print('X', x)
print('Y', y)

mem_x = brisbane.mem(x.nbytes)
mem_y = brisbane.mem(y.nbytes)
mem_s = brisbane.mem(s.nbytes)

task = brisbane.task()
task.h2d_full(mem_x, x)
task.h2d_full(mem_y, y)
task.kernel("saxpy", 1, [], [SIZE], [], [mem_s, A, mem_x, mem_y] , [brisbane.brisbane_w, 4, brisbane.brisbane_r, brisbane.brisbane_r] )
task.d2h_full(mem_s, s)
task.submit(brisbane.brisbane_gpu)

print('S =', A, '* X + Y', s)

brisbane.finalize()

