#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE;
  int TARGET;
  int VERBOSE;
  float *X, *Y, *Z;
  float A = 1;
  int ERROR = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  VERBOSE = argc > 3 ? atol(argv[3]) : 1;

  printf("[%s:%d] SIZE[%zu] TARGET[%d] VERBOSE[%d]\n", __FILE__, __LINE__, SIZE, TARGET, VERBOSE);

  X = (float*) malloc(SIZE * sizeof(float));
  Y = (float*) malloc(SIZE * sizeof(float));
  Z = (float*) malloc(SIZE * sizeof(float));

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    X[i] = 1;
    Y[i] = 2;
    Z[i] = 0;
  }

  printf("X [");
  for (int i = 0; i < SIZE; i++) printf(" %2.0f.", X[i]);
  printf("]\n");
  printf("Y [");
  for (int i = 0; i < SIZE; i++) printf(" %2.0f.", Y[i]);
  printf("]\n");

  }

  iris_graph graph;
  iris_graph_create(&graph);

  iris_mem mem_X;
  iris_mem mem_Y;
  iris_mem mem_Z;
  iris_data_mem_create(&mem_X, X, SIZE * sizeof(float));
  iris_data_mem_create(&mem_Y, Y, SIZE * sizeof(float));
  iris_data_mem_create(&mem_Z, Z, SIZE * sizeof(float));
  if (mem_Z == NULL) printf("mem is nnull \n");
  char tn[128]; 
  sprintf(tn, "task0", NULL);
 
  iris_task task0;
  //iris_task_create(&task0);
  iris_task_create_name(tn, &task0); 
  void* params0[4] = { mem_Z, &A, mem_Y, mem_Y };
  int pinfo0[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task0, "saxpy", 1, NULL, &SIZE, NULL, 4, params0, pinfo0);
  //iris_task_dmem_flush_out(task0, mem_Z);
  iris_graph_task(graph, task0, iris_any, NULL);
  //iris_task_submit(task0, iris_any, nullptr, true);


  sprintf(tn, "task1", NULL);
  iris_task task1;
  //iris_task_create(&task1);
  iris_task_create_name(tn, &task1); 
  void* params1[4] = { mem_X, &A, mem_X, mem_Z };
  int pinfo1[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task1, "saxpy", 1, NULL, &SIZE, NULL, 4, params1, pinfo1);
  //iris_task_dmem_flush_out(task1,mem_X);
  iris_graph_task(graph, task1, iris_any, NULL);
  //iris_task_submit(task1, iris_any, nullptr, true);

  sprintf(tn, "task2", NULL);
  iris_task task2;
  //iris_task_create(&task2);
  iris_task_create_name(tn, &task2); 
  void* params2[4] = { mem_Z, &A, mem_Z, mem_Y };
  int pinfo2[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task2, "saxpy", 1, NULL, &SIZE, NULL, 4, params2, pinfo2);
  //iris_task_dmem_flush_out(task2, mem_Y);
  //iris_task_submit(task2, iris_any, nullptr, true);
  iris_graph_task(graph, task2, iris_any, NULL);

  sprintf(tn, "task3", NULL);
  iris_task task3;
  //iris_task_create(&task1);
  iris_task_create_name(tn, &task3); 
  void* params3[4] = { mem_X, &A, mem_X, mem_Z };
  int pinfo3[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task3, "saxpy", 1, NULL, &SIZE, NULL, 4, params3, pinfo3);
  //iris_task_dmem_flush_out(task1,mem_X);
  iris_graph_task(graph, task3, iris_any, NULL);
  //iris_task_submit(task1, iris_any, nullptr, true);

  sprintf(tn, "task4", NULL);
  iris_task task4;
  //iris_task_create(&task2);
  iris_task_create_name(tn, &task4); 
  void* params4[4] = { mem_Z, &A, mem_Z, mem_Y };
  int pinfo4[4] = { iris_w, sizeof(A), iris_r, iris_r };
  iris_task_kernel(task4, "saxpy", 1, NULL, &SIZE, NULL, 4, params4, pinfo4);
  //iris_task_dmem_flush_out(task2, mem_Y);
  //iris_task_submit(task2, iris_any, nullptr, true);
  iris_graph_task(graph, task4, iris_any, NULL);

  printf("Graph creation done\n");
  iris_graph_submit(graph, iris_any, 1);

  iris_synchronize();

  if (VERBOSE) {

  for (int i = 0; i < SIZE; i++) {
    //printf("[%8d] %8.1f = %4.0f * %8.1f + %8.1f\n", i, Z[i], A, X[i], Y[i]);
    if (Z[i] != A * X[i] + Y[i]) ERROR++;
  }

  printf("Z = %p\n", Z);
  printf("Z = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", Z[i]);
  printf("]\n");

  printf("X = %p\n", X);
  printf("X = [");
  //printf("S = %f * X + Y [", A);
  for (int i = 0; i < SIZE; i++) printf(" %3.0f.", X[i]);
  printf("]\n");

  }

  iris_mem_release(mem_X);
  iris_mem_release(mem_Y);
  iris_mem_release(mem_Z);

  free(X);
  free(Y);
  free(Z);

  //iris_task_release(task0);

  iris_finalize();

  return 0;
}
