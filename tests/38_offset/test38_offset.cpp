#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

//this test assesses the use of data memory with the old explicit kernel object creation and setting memory arguments vs the newer way (iris_task_kernel)
int main(int argc, char** argv) {
  size_t SIZE;
  int *A, *B, *C;
  int ERROR = 0;

  iris_init(&argc, &argv, true);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  printf("SIZE[%lu]\n", SIZE);

  A = (int*) valloc(SIZE * sizeof(int));
  B = (int*) valloc(SIZE * sizeof(int));
  C = (int*) valloc(SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i;
    C[i] = 0;
  }

  iris_mem mem_A;
  iris_mem mem_B;
  iris_mem mem_C;
  iris_data_mem_create(&mem_A, A, SIZE * sizeof(int));
  iris_data_mem_create(&mem_B, B, SIZE * sizeof(int));
  iris_data_mem_create(&mem_C, C, SIZE * sizeof(int));

  size_t byte_offset = sizeof(int)*SIZE/2;// leave the first half of the results
                                          // the offset is specified in bytes
  size_t NUM_WORK_ITEMS = SIZE/2; //we'll only operate on half the data in this offset test, so we have half the total number of work-items

#if 0
  // test memory offset (explicit)
  iris_kernel kernel0;
  iris_kernel_create("vecadd", &kernel0);

  iris_kernel_setmem_off(kernel0, 0, mem_A, byte_offset, iris_r);
  iris_kernel_setmem_off(kernel0, 1, mem_B, byte_offset, iris_r);
  iris_kernel_setmem_off(kernel0, 2, mem_C, byte_offset, iris_rw);
  
  iris_task task0;
  iris_task_create(&task0);
  iris_task_kernel_object(task0, kernel0, 1, NULL, &NUM_WORK_ITEMS, NULL);
  
  iris_task_dmem_flush_out(task0,mem_C);
  iris_task_submit(task0, iris_sdq, nullptr, true);
  iris_synchronize();

  for (int i = 0; i < SIZE; i++) {
    printf("C[%d] = %d\n", i, C[i]);
    if (i < SIZE/2) {
      if (C[i] != 0) ERROR++;
    }
    else {
      if (C[i] != (A[i] + B[i])) ERROR++;
    }
    C[i] = 0;
  }
  printf("ERROR[%d]\n", ERROR+iris_error_count());
  iris_data_mem_update(mem_C, C);
#endif
  size_t OFFSET = SIZE/2;
  // test memory offset (data-memory)
  iris_task task1;
  iris_task_create(&task1);

  void* params[3] = { &mem_A, &mem_B, &mem_C };
  int pinfo[3] = { iris_r, iris_r, iris_rw };
  size_t poff[1] = { OFFSET };
  iris_task_kernel(task1, "vecadd", 1, poff, &NUM_WORK_ITEMS, NULL, 3, params, pinfo);

  iris_task_dmem_flush_out(task1,mem_C);
  iris_task_submit(task1, iris_sdq, nullptr, true);
  iris_synchronize();

  for (int i = 0; i < SIZE; i++) {
    //printf("C[%d] = %d\n", i, C[i]);
    if (i < SIZE/2) {
      if (C[i] != 0) ERROR++;
    }
    else {
      if (C[i] != (A[i] + B[i])) ERROR++;
    }
    printf("C[%d]:%d A[%d]:%d B[%d]:%d\n", i, C[i], i, A[i], i, B[i]);
  }
  printf("ERROR[%d]\n", ERROR+iris_error_count());

  // 2-D tests
  iris_mem_release(mem_A);
  iris_mem_release(mem_B);
  iris_mem_release(mem_C);
  free(A);
  free(B);
  free(C);
  A = (int*) valloc(SIZE * SIZE * sizeof(int));
  B = (int*) valloc(SIZE * SIZE * sizeof(int));
  C = (int*) valloc(SIZE * SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      A[j * SIZE + i] = i;
      B[j * SIZE + i] = j;
      C[j * SIZE + i] = 0;
    }
  }
  
  iris_data_mem_create(&mem_A, A, SIZE * SIZE * sizeof(int));
  iris_data_mem_create(&mem_B, B, SIZE * SIZE * sizeof(int));
  iris_data_mem_create(&mem_C, C, SIZE * SIZE * sizeof(int));

  iris_task task2;
  iris_task_create(&task2);

  void* block_params[] = { &mem_A, &mem_B, &mem_C, (void *)&SIZE };
  int block_pinfo[] = { iris_r, iris_r, iris_rw, sizeof(SIZE) };
  size_t gworksize[2] = {SIZE, SIZE};
  size_t worksize[2] = {NUM_WORK_ITEMS, NUM_WORK_ITEMS};
  size_t block_off[2] = {OFFSET, OFFSET};
  iris_task_kernel(task2, "blockadd", 2, block_off, worksize, NULL, 4, block_params, block_pinfo);

  iris_task_dmem_flush_out(task2, mem_C);
  iris_task_submit(task2, iris_sdq, nullptr, true);
  iris_synchronize();

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (i < SIZE/2 || j < SIZE/2) {
        if (C[j * SIZE + i] != 0) {printf("incrementing error here!\n");ERROR++;}
      }
      else {
        if (C[j * SIZE + i] != (A[j * SIZE + i] + B[j * SIZE + i])) ERROR++;
      }
      printf("C[%d]:%d A[%d]:%d B[%d]:%d\n", j * SIZE + i, C[j * SIZE + i], j * SIZE + i, A[j * SIZE + i], j * SIZE + i, B[j * SIZE + i]);
    }
  }

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      printf("%5d ", C[i*SIZE+j]);
    }
    printf("\n");
  }
  printf("\n");

  // TODO: 3-D tests

  iris_finalize();
  printf("ERROR[%d]\n", ERROR+iris_error_count());
  return ERROR+iris_error_count();
}
