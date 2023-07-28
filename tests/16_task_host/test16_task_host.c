#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

size_t SIZE;

int getFactor(void* argp) {
  printf("input factor: ");
  scanf("%d", (int*) argp);
  return 0;
}

int printA(void* argp) {
  int* A = (int*) argp;
  for (int i = 0; i < SIZE; i++) printf("[%3d] %8d\n", i, A[i]);
  return 0;
}

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  int LOOP;
  int* A;
  int factor;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  LOOP = argc > 2 ? atoi(argv[2]) : 3;

  printf("[%s:%d] SIZE[%lu] LOOP[%d] \n", __FILE__, __LINE__, SIZE, LOOP);

  A = (int*) malloc(SIZE * sizeof(int));

  iris_graph graph;
  iris_graph_create(&graph);

  iris_mem memA;
  iris_mem memFactor;
  iris_mem_create(SIZE * sizeof(int), &memA);
  iris_mem_create(sizeof(int), &memFactor);

  iris_task task1;
  iris_task_create(&task1);
  iris_task_host(task1, getFactor, &factor);
  iris_graph_task(graph, task1, iris_any, NULL);

  void* params[2] = { &memA, &memFactor };
  int params_info[2] = { iris_w, iris_r };
  iris_task task2;
  iris_task_create(&task2);
  iris_task_h2d_full(task2, memFactor, &factor);
  iris_task_kernel(task2, "process", 1, NULL, &SIZE, NULL, 2, params, params_info);
  iris_task_d2h_full(task2, memA, A);
  iris_graph_task(graph, task2, iris_any, NULL);

  iris_task task3;
  iris_task_create(&task3);
  iris_task_host(task3, printA, A);
  iris_graph_task(graph, task3, iris_any, NULL);

  iris_graph_retain(graph, true);
  for (int loop = 0; loop < LOOP; loop++)
    iris_graph_submit(graph, iris_any, 0);

  iris_finalize();

  return iris_error_count();
}

