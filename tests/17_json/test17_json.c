#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  size_t SIZE, SIZECB;
  int *A, *B;
  int target = iris_gpu;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  SIZECB = SIZE * sizeof(int);

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));

  iris_mem mem;
  iris_mem_create(SIZE * sizeof(int), &mem);

  void* json_inputs[6] = { &SIZE, &SIZECB, A, B, mem, &target };

  iris_graph graph;
  iris_graph_create_json("graph.json", json_inputs, &graph);

  for (int i = 0; i < SIZE; i++) A[i] = i;
  iris_graph_submit(graph, iris_default, true);
  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  iris_finalize();

  return 0;
}

