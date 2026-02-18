#!/usr/bin/env python3
import iris
import numpy as np
import sys
import pdb

iris.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

@iris.kernel(in_=["X", "Y"], flush_=["Z"])
def py_saxpy(spec, Z, alpha, Y, X):
    Z[:] = alpha * X + Y

@iris.kernel(in_=["X", "Y"], flush_=["Z"])
def saxpy(spec, Z, alpha, Y, X):
    pass

x = np.arange(SIZE, dtype=np.float32)
y = np.arange(SIZE, dtype=np.float32)
s = np.arange(SIZE, dtype=np.float32)

task0 = py_saxpy(s, A, x, y, gws=[SIZE])
task0.submit()

task1 = saxpy(s, A, x, y, gws=[SIZE])
task1.submit()

print('S =', A, '* X + Y', 'A:', A, 'X:',x, 'Y:', y, 'S:', s)


iris.finalize()

