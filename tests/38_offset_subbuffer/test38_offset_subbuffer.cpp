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

#if 1
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

  iris_finalize();
  printf("ERROR[%d]\n", ERROR+iris_error_count());
  return ERROR+iris_error_count();
}
