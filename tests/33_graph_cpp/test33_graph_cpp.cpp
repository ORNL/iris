#include <iris/iris.hpp>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, true);

  size_t SIZE;
  int *A, *B;

  SIZE = argc > 1 ? atol(argv[1]) : 8;

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));

  iris::Mem mem(SIZE * sizeof(int));
  iris::Graph graph;
  graph.retainable();

  iris::Task task1("task1", true);
  task1.h2d_full(&mem, A);
  graph.add_task(task1, iris_default);

  iris::Task task2("task2", true);
  task2.depends_on(task1);
  void* params[1] = { &mem };
  int params_info[1] = { iris_rw };
  task2.kernel("process", 1, NULL, &SIZE, NULL, 1, params, params_info);
  graph.add_task(task2, iris_default);

  iris::Task task3("task3", true);
  task3.depends_on(task2);
  task3.d2h_full(&mem, B);
  graph.add_task(task3, iris_data);

  for (int i = 0; i < SIZE; i++) A[i] = i;
  graph.submit(iris_cpu, false);
  graph.wait();
  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  for (int i = 0; i < SIZE; i++) A[i] = i * 10;
  graph.submit(iris_gpu, false);
  graph.wait();
  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  platform.finalize();

  return platform.error_count();
}

