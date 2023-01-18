#!/usr/bin/env python

import iris

iris.init(True)

task4 = iris.task_create("A")
iris.task_submit(task4, iris.iris_cpu, False);

task5 = iris.task_create("B")
iris.task_submit(task5, iris.iris_gpu, False);

task6 = iris.task_create("C")
task6_dep = [ task5 ]
iris.task_depend(task6, 1, task6_dep);
iris.task_submit(task6, iris.iris_cpu, False);

task7 = iris.task_create("D")
task7_dep = [ task4, task6 ]
iris.task_depend(task7, 2, task7_dep);
iris.task_submit(task7, iris.iris_gpu, False);

task8 = iris.task_create("E")
task8_dep = [ task5 ]
iris.task_depend(task8, 1, task8_dep);
iris.task_submit(task8, iris.iris_cpu, False);

iris.finalize()

