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

  for (int i = 0; i < SIZE; i++) A[i] = i;

  iris_mem mem;
  iris_mem_create(SIZE * sizeof(int), &mem);

  void* json_inputs[3] = { A, B, &mem };

  iris_graph graph;
  iris_graph_create_json("../18_record/output.json", json_inputs, &graph);
  iris_graph_submit(graph, iris_default, true);

  for (int i = 0; i < SIZE; i++) printf("[%3d] %3d\n", i, B[i]);

  iris_finalize();

  return iris_error_count();
}

