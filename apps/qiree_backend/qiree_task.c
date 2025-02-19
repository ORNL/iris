#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE = 0;
  int TARGET;
  int VERBOSE;

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  VERBOSE = argc > 3 ? atol(argv[3]) : 1;

  printf("[%s:%d] SIZE[%zu] TARGET[%d] VERBOSE[%d]\n", __FILE__, __LINE__, SIZE, TARGET, VERBOSE);

  //iris_graph graph;
  //iris_graph_create(&graph);

  char tn[128]; 
  sprintf(tn, "qiree_task", NULL);
 
  iris_task task0;
  iris_task_create_name(tn, &task0); 
  //void* params0[] = {};
  //int pinfo0[] = {};
  iris_task_kernel(task0, "bell.ll", 1, NULL, &SIZE, NULL, 0, NULL, NULL);
  //iris_task_kernel(task0, "saxpy", 1, NULL, &SIZE, NULL, 4, params0, pinfo0);
  //iris_task_dmem_flush_out(task0, mem_Z);
  //iris_graph_task(graph, task0, iris_any, NULL);
  iris_task_submit(task0, iris_cpu, NULL, 1);

  iris_synchronize();

  iris_finalize();

  return 0;
}
