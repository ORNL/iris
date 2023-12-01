#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

size_t SIZE;

int getFactor(void* argp) {
  printf("input factor: ");
  *((int *)argp) = 2;
  //scanf("%d", (int*) argp);
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
  printf("Factor: %p\n", &factor);

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
  iris_task_create_name("task1", &task1);
  iris_task_host(task1, getFactor, &factor);
  iris_graph_task(graph, task1, iris_sdq, NULL);

  void* params[2] = { &memA, &memFactor };
  int params_info[2] = { iris_w, iris_r };
  iris_task task2;
  iris_task_create_name("task2", &task2);
  iris_task_h2d_full(task2, memFactor, &factor);
  iris_task_kernel(task2, "process", 1, NULL, &SIZE, NULL, 2, params, params_info);
  iris_task_d2h_full(task2, memA, A);
  iris_task_depend(task2, 1, &task1);
  iris_graph_task(graph, task2, iris_sdq, NULL);
  iris_task task3;
  iris_task_create_name("task3", &task3);
  iris_task_host(task3, printA, A);
  iris_task_depend(task3, 1, &task2);
  iris_graph_task(graph, task3, iris_sdq, NULL);

  iris_graph_retain(graph, true);
  int errors = 0;
  for (int loop = 0; loop < LOOP; loop++) {
    factor = loop+1;
    iris_graph_submit(graph, iris_sdq, 0);
    iris_synchronize();
    for(int i=0; i<SIZE; i++) {
        if (A[i] != i * factor) 
            errors++;
    }
  }
  iris_graph_release(graph);
  iris_finalize();
  
  errors += iris_error_count();
  printf("N-Errors: %d\n", errors);
  return errors;
}

