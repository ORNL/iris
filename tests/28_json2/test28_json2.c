#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE, SIZECB;
  int *A, *B, *C;
  int target1 = iris_roundrobin;
  int target2 = iris_depend;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  SIZECB = SIZE * sizeof(int);

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 10;
    C[i] = 0;
  }

  iris_mem memA;
  iris_mem memB;
  iris_mem memC;
  iris_mem_create(SIZE * sizeof(int), &memA);
  iris_mem_create(SIZE * sizeof(int), &memB);
  iris_mem_create(SIZE * sizeof(int), &memC);

  void* json_inputs[10] = { &SIZE, &SIZECB, C, A, B, &memC, &memA, &memB, &target1, &target2 };

  iris_graph graph;
  iris_graph_create_json("graph.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_any, 1);

  iris_synchronize();
  int errs = 0;
  for (int i = 0; i < SIZE; i++) {
    if (C[i] != (A[i] + B[i])*10)//for ten iterations (see graph.json for the number of ijk kernels invoked)
      errs++;
    printf("[%3d] %3d\n", i, C[i]);
  }

  iris_finalize();

  return iris_error_count()+errs;
}

