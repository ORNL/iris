#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  iris_task task2;
  iris_task_create(&task2);
  iris_task_submit(task2, iris_cpu, NULL, false);

  iris_task task3;
  iris_task_create(&task3);
  iris_task_submit(task3, iris_gpu, NULL, false);

  iris_task task4;
  iris_task task4_dep[] = { task3 };
  iris_task_create(&task4);
  iris_task_depend(task4, 1, task4_dep);
  iris_task_submit(task4, iris_cpu, NULL, false);

  iris_task task5;
  iris_task task5_dep[] = { task2, task4 };
  iris_task_create(&task5);
  iris_task_depend(task5, 2, task5_dep);
  iris_task_submit(task5, iris_gpu, NULL, false);

  iris_task task6;
  iris_task task6_dep[] = { task2 };
  iris_task_create(&task6);
  iris_task_depend(task6, 1, task6_dep);
  iris_task_submit(task6, iris_cpu, NULL, false);

  iris_finalize();

  return iris_error_count();
}
