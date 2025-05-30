#!/usr/bin/env python3

import iris
import numpy as np
import sys

iris.init()

SIZE = 8 if len(sys.argv) == 1 else int(sys.argv[1])
A = 10.0

n = 64
tasks = [iris.task() for i in range(n)]
#task = iris.task()

for i in range(n):
    tasks[i].kernel("bell.ll", 1, [], [SIZE], [], [] , [] )
    tasks[i].submit(iris.iris_default, sync=0)

iris.finalize()

