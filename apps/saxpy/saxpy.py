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

old_way = True
disable = True
if not disable:
    if old_way:
        mem_x = iris.mem(x.nbytes)
        mem_y = iris.mem(y.nbytes)
        mem_s = iris.mem(s.nbytes)

        task = iris.task()
        task.h2d_full(mem_x, x)
        task.h2d_full(mem_y, y)
        task.kernel("saxpy", 1, [], [SIZE], [], [mem_s, A, mem_x, mem_y] , [iris.iris_w, 4, iris.iris_r, iris.iris_r] )
        task.d2h_full(mem_s, s)
        task.submit(iris.iris_gpu)
    else:
        # New DMEM way
        task = iris.task("saxpy", 1, [], [SIZE], [], [
                (s, iris.iris_w, iris.iris_flush),
                A, 
                (x, iris.iris_r), 
                (y, iris.iris_r)
                ])
        task.submit(iris.iris_gpu)
    
print('S =', A, '* X + Y', s)

s0 = np.arange(SIZE, dtype=np.float32)
s1 = np.arange(SIZE, dtype=np.float32)

mem_x = iris.dmem(x)
mem_y = iris.dmem(y)
#mem_s0 = iris.dmem(s0)
mem_s0 = iris.dmem_null(s0.nbytes)
mem_s1 = iris.dmem(s1)
task0 = iris.task("saxpy", 1, [], [SIZE], [], [
        (mem_s0, iris.iris_w),
        A, 
        (mem_x, iris.iris_r), 
        (mem_y, iris.iris_r)
        ])
task1 = iris.task("saxpy", 1, [], [SIZE], [], [
        (mem_s1, iris.iris_w, iris.iris_flush),
        A, 
        (mem_s0, iris.iris_r), 
        (mem_s0, iris.iris_r)
        ])
task1.depends(task0)
#task0.submit(iris.iris_gpu)
#task1.submit(iris.iris_gpu)

#print('S =', A, '* X + Y', x, y, s0)
#print('S =', A, '* X + Y', s0, s0, s1)

x = x*100
mem_x.update(x)

graph = iris.graph([task0, task1])
graph.submit()
graph.wait()

print('S =', A, '* X + Y', x, y, s0)
print('S =', A, '* X + Y', s0, s0, s1)

x = x*100
mem_x.update(x)
graph.submit()
graph.wait()

ntasks, tasks = graph.get_tasks()
print('S =', A, '* X + Y', x, y, s0)
print('S =', A, '* X + Y', s0, s0, s1)



iris.finalize()

