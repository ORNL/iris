#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

size_t SIZE = 8;
int *A, *B, *C;
int ncmds, ncmds_kernel, ncmds_memcpy;

iris_task* tasks;

int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  time_t t;
  srand((unsigned int) time(&t));

  A = (int*) malloc(SIZE * sizeof(int));
  B = (int*) malloc(SIZE * sizeof(int));
  C = (int*) malloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i * 100;
  }

  printf("[%s:%d] A [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", A[i]);
  printf("]\n");

  iris_mem mem_A, mem_B, mem_C;
  iris_mem_create(SIZE * sizeof(int), &mem_A);
  iris_mem_create(SIZE * sizeof(int), &mem_B);
  iris_mem_create(SIZE * sizeof(int), &mem_C);

  iris_task task0;
  iris_task_create(&task0);
  iris_task_h2d_full(task0, mem_A, A);
  iris_task_info(task0, iris_ncmds, &ncmds, NULL);
  iris_task_info(task0, iris_ncmds_kernel, &ncmds_kernel, NULL);
  iris_task_info(task0, iris_ncmds_memcpy, &ncmds_memcpy, NULL);
  printf("[%s:%d] ncmds[%d] kernel[%d] memcpy[%d]\n", __FILE__, __LINE__, ncmds, ncmds_kernel, ncmds_memcpy);
  iris_task_submit(task0, iris_roundrobin, NULL, 1);

  iris_task task1;
  iris_task_create(&task1);
  iris_task_h2d_full(task1, mem_B, B);
  iris_task_info(task1, iris_ncmds, &ncmds, NULL);
  iris_task_info(task1, iris_ncmds_kernel, &ncmds_kernel, NULL);
  iris_task_info(task1, iris_ncmds_memcpy, &ncmds_memcpy, NULL);
  printf("[%s:%d] ncmds[%d] kernel[%d] memcpy[%d]\n", __FILE__, __LINE__, ncmds, ncmds_kernel, ncmds_memcpy);
  iris_task_submit(task1, iris_roundrobin, NULL, 1);

  iris_task task2;
  iris_task_create(&task2);
  void* params[3] = { &mem_C, &mem_A, &mem_B };
  int params_info[3] = { iris_w, iris_r, iris_r };
  iris_task_kernel(task2, "vecadd", 1, NULL, &SIZE, NULL, 3, params, params_info);
  iris_task_d2h_full(task2, mem_C, C);
  iris_task_info(task2, iris_ncmds, &ncmds, NULL);
  iris_task_info(task2, iris_ncmds_kernel, &ncmds_kernel, NULL);
  iris_task_info(task2, iris_ncmds_memcpy, &ncmds_memcpy, NULL);
  printf("[%s:%d] ncmds[%d] kernel[%d] memcpy[%d]\n", __FILE__, __LINE__, ncmds, ncmds_kernel, ncmds_memcpy);
  /* please see iris-rts/src/runtime/Command.h
#define IRIS_CMD_NOP            0x1000
#define IRIS_CMD_INIT           0x1001
#define IRIS_CMD_KERNEL         0x1002
#define IRIS_CMD_MALLOC         0x1003
#define IRIS_CMD_H2D            0x1004
#define IRIS_CMD_H2DNP          0x1005
#define IRIS_CMD_D2H            0x1006
#define IRIS_CMD_MAP            0x1007
#define IRIS_CMD_MAP_TO         0x1008
#define IRIS_CMD_MAP_FROM       0x1009
#define IRIS_CMD_RELEASE_MEM    0x100a
#define IRIS_CMD_HOST           0x100b
#define IRIS_CMD_CUSTOM         0x100c
   */
  int* cmds = (int*) malloc(ncmds * sizeof(int));
  iris_task_info(task2, iris_cmds, cmds , NULL);
  for (int i = 0; i < ncmds; i++) {
    int type = cmds[i];
    printf("[%s:%d] cmd[%d] type[%x]\n", __FILE__, __LINE__, i, type);
  }
  free(cmds);
  iris_task_submit(task2, iris_roundrobin, NULL, 1);

  iris_task task3;
  iris_task_create(&task3);
  iris_task_d2h_full(task3, mem_C, C);
  iris_task_info(task3, iris_ncmds, &ncmds, NULL);
  iris_task_info(task3, iris_ncmds_kernel, &ncmds_kernel, NULL);
  iris_task_info(task3, iris_ncmds_memcpy, &ncmds_memcpy, NULL);
  printf("[%s:%d] ncmds[%d] kernel[%d] memcpy[%d]\n", __FILE__, __LINE__, ncmds, ncmds_kernel, ncmds_memcpy);
  iris_task_submit(task3, iris_roundrobin, NULL, 1);

  printf("[%s:%d] A [", __FILE__, __LINE__);
  for (int i = 0; i < SIZE; i++) printf(" %4d", C[i]);
  printf("]\n");

  iris_mem_release(mem_A);
  iris_mem_release(mem_B);
  iris_mem_release(mem_C);

  iris_finalize();

  free(A);
  free(B);
  free(C);
  free(tasks);

  return iris_error_count();
}
