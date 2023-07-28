#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  size_t SIZE, SIZECB;
  int *A, *B, *C;
  int target = iris_default;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  SIZECB = SIZE * sizeof(int);

  printf("[%s:%d] SIZE[%lu]\n", __FILE__, __LINE__, SIZE);

  //initialize memory
  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = 0;
    C[i] = 0;
  }
  //create and pass iris memory
  iris_mem mem;
  iris_mem_create(SIZE * sizeof(int), &mem);

  void* json_inputs[5] = { &SIZE, &SIZECB, B, &mem, &target };

  iris_graph graph;
  iris_graph_create_json("graph.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_any, true);
  iris_synchronize();
  iris_graph_free(graph);
  int errs = 0;
  for (int i = 0; i < SIZE; i++) {
    if (A[i] != B[i]) errs++;
  }
  if (errs != 0)
    exit(errs);
  printf("Passed the portion of the test using explicit iris memory\n");
  //create and pass iris data memory
  iris_mem dmem;
  iris_data_mem_create(&dmem,C,SIZE*sizeof(int));
  iris_graph dmemgraph;
  void* dmem_json_inputs[5] = { &SIZE, &SIZECB, C, &dmem, &target };
  int retval = iris_graph_create_json("graph.json", dmem_json_inputs, &dmemgraph);
  assert(retval == IRIS_SUCCESS);
  iris_graph_submit(dmemgraph, iris_any, true);
  iris_synchronize();

  for (int i = 0; i < SIZE; i++) {
    printf("[%3d] A:%3d B:%3d C:%3d\n", i, A[i],B[i],C[i]);
    if (A[i] != C[i]) errs++;
  }

  iris_finalize();

  printf("return code = %i\n",iris_error_count()+errs);

  return iris_error_count()+errs;
}

