#!/usr/bin/env python3

import iris
import numpy as np
import sys

iris.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
s0 = np.arange(SIZE, dtype=np.float32)

print('X', x)
print('Y', y)

task0 = iris.task("runtime_kernel:saxpy", 1, [], [SIZE], [], [
        (s0, iris.iris_w, iris.iris_flush),
        A, 
        (x, iris.iris_r), 
        (y, iris.iris_r)
        ])
task0.submit(iris.iris_default)

print('S0 =', A, '* X + Y', 'A:', A, 'x:', x, 'y:', y, 'S0:',s0)

iris.finalize()

