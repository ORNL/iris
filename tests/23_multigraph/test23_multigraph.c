#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

size_t SIZE;
int NGRAPHS;
int LOOP;
double *A, *B, *C;
double t0, t1;

iris_mem* mem_A;
iris_mem* mem_B;
iris_mem* mem_C;
iris_graph* graph;


double now() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1.e-9 * t.tv_nsec;
}

void build_graph(iris_mem mem_A, iris_mem mem_B, iris_mem mem_C, iris_graph G) {
  iris_task task1;
  iris_task_create(&task1);
  iris_task_h2d_full(task1, mem_A, A);
  iris_task_h2d_full(task1, mem_B, B);
  iris_graph_task(G, task1, iris_default, NULL);

  size_t gws[2] = { SIZE, SIZE };
  size_t lws[2] = { 1, 1 };
  void* params[3] = { &mem_C, &mem_A, &mem_B };
  int pinfo[3] = { iris_w, iris_r, iris_r };
  iris_task task2;
  iris_task task2_dep[] = { task1 };
  iris_task_create(&task2);
  iris_task_depend(task2, 1, task2_dep);
  iris_task_kernel(task2, "ijk", 2, NULL, gws, lws, 3, params, pinfo);
  iris_graph_task(G, task2, iris_depend, NULL);

  iris_task task3;
  iris_task task3_dep[] = { task2 };
  iris_task_create(&task3);
  iris_task_depend(task3, 1, task3_dep);
  iris_task_d2h_full(task3, mem_C, C);
  iris_graph_task(G, task3, iris_depend, NULL);
}

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  NGRAPHS = argc > 2 ? atoi(argv[2]) : 1;
  LOOP = argc > 3 ? atoi(argv[3]) : 1;

  printf("[%s:%d] SIZE[%lu] NGRAPHS[%d] LOOP[%d]\n", __FILE__, __LINE__, SIZE, NGRAPHS, LOOP);

  A = (double*) malloc(SIZE * SIZE * sizeof(double));
  B = (double*) malloc(SIZE * SIZE * sizeof(double));
  C = (double*) malloc(SIZE * SIZE * sizeof(double));

  mem_A = (iris_mem*) malloc(sizeof(iris_mem) * NGRAPHS);
  mem_B = (iris_mem*) malloc(sizeof(iris_mem) * NGRAPHS);
  mem_C = (iris_mem*) malloc(sizeof(iris_mem) * NGRAPHS);
  graph = (iris_graph*) malloc(sizeof(iris_graph) * NGRAPHS);

  for (int i = 0; i < NGRAPHS; i++) {
    iris_mem_create(SIZE * SIZE * sizeof(double), mem_A + i);
    iris_mem_create(SIZE * SIZE * sizeof(double), mem_B + i);
    iris_mem_create(SIZE * SIZE * sizeof(double), mem_C + i);
    iris_graph_create(graph + i);
    build_graph(mem_A[i], mem_B[i], mem_C[i], graph[i]);
  }

  for (int loop = 0; loop < LOOP; loop++) {
    printf("LOOP [%d] START \n", loop);
    t0 = now();
    for (int i = 0; i < NGRAPHS; i++) {
      iris_graph_submit(graph[i], iris_roundrobin | iris_gpu, 0);
    }
    iris_synchronize();
    t1 = now();
    printf("LOOP [%d] END %lf secs\n", loop, t1 - t0);
  }

  iris_finalize();

  return iris_error_count();
}

