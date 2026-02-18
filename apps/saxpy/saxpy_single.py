#!/usr/bin/env python3

import iris
import numpy as np
import sys
import pdb

iris.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

@iris.ptask(in_=["X", "Y"], flush_=["Z"])
def saxpy(spec, Z, alpha, Y, X):
    Z[:] = alpha * X + Y

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
s = np.arange(SIZE, dtype=np.float32)

print('X', x)
print('Y', y)
# New DMEM way
original = False
if original:
    task0 = iris.task(saxpy, 1, [], [SIZE], [], [
        (s, iris.iris_w, iris.iris_flush),
        A, 
        (x, iris.iris_r), 
        (y, iris.iris_r)
        ])
else:
    task0 = saxpy(s, A, x, y, gws=[SIZE])
task0.submit(iris.iris_gpu)
iris.sync()

print('S =', A, '* X + Y', 'A:', A, 'X:',x, 'Y:', y, 'S:', s)


iris.finalize()

