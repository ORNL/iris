#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  size_t SIZE;
  int *A, *B;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));

  iris_mem mem;
  iris_mem_create(SIZE * sizeof(int), &mem);

  iris_graph graph;
  iris_graph_create(&graph);

  iris_task task1;
  iris_task_create(&task1);
  iris_task_set_name(task1, "task1");

  iris_task_h2d_full(task1, mem, A);
  iris_graph_task(graph, task1, iris_default, NULL);

  void* params[1] = { &mem };
  int params_info[1] = { iris_rw };
  iris_task task2;
  iris_task task2_dep[] = { task1 };
  iris_task_create(&task2);
  iris_task_set_name(task2, "task2");
  iris_task_depend(task2, 1, task2_dep);
  iris_task_kernel(task2, "process", 1, NULL, &SIZE, NULL, 1, params, params_info);
  iris_graph_task(graph, task2, iris_default, NULL);

  iris_task task3;
  iris_task task3_dep[] = { task2 };
  iris_task_create(&task3);
  iris_task_set_name(task3, "task3");
  iris_task_depend(task3, 1, task3_dep);
  iris_task_d2h_full(task3, mem, B);
  iris_graph_task(graph, task3, iris_data, NULL);

  for (int i = 0; i < SIZE; i++) A[i] = i;
  iris_graph_retain(graph, true);
  iris_graph_submit(graph, iris_cpu, false);
  iris_graph_wait(graph);
  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  for (int i = 0; i < SIZE; i++) A[i] = i * 10;
  iris_graph_submit(graph, iris_gpu, false);
  iris_graph_wait(graph);
  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  iris_graph_release(graph);
  iris_finalize();

  printf("Nerrors:%d\n", iris_error_count());
  return iris_error_count();
}

