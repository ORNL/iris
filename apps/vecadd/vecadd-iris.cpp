#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <iris/iris.h>

int main(int argc, char** argv) {
  size_t SIZE;
  size_t REPEATS;
  bool VERIFY;
  const char* LOG_FILENAME;
  int *A, *B, *C;
  int ERROR_COUNT = 0;

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  REPEATS = argc > 2 ? atol(argv[2]) : 1;
  VERIFY = REPEATS == 1 ? true : false;
  LOG_FILENAME = argc > 3 ? argv[3] : "timings.csv";
  std::vector<std::string> timings;
  printf("SIZE[%lu]\n", SIZE);

  iris_init(&argc, &argv, true);
  char info[256];
  size_t size;
  iris_device_info(0, iris_name, &info, &size);
  printf("Running on %s\n",info);

  for (int reps = 0; reps < REPEATS; reps++){
    A = (int*) valloc(SIZE * sizeof(int));
    B = (int*) valloc(SIZE * sizeof(int));
    C = (int*) valloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; i++) {
      A[i] = i;
      B[i] = i;
      C[i] = 0;
    }

    auto const t_start = std::chrono::high_resolution_clock::now();
    iris_mem mem_A;
    iris_mem mem_B;
    iris_mem mem_C;
    iris_mem_create(SIZE * sizeof(int), &mem_A);
    iris_mem_create(SIZE * sizeof(int), &mem_B);
    iris_mem_create(SIZE * sizeof(int), &mem_C);

    /*
    //another way of creating the kernel and associating it to the task
    iris_kernel vecadd;
    iris_kernel_create("vecadd", &vecadd);
    iris_kernel_setmem(vecadd, 0 ,mem_A, iris_r);
    iris_kernel_setmem(vecadd, 1 ,mem_B, iris_r);
    iris_kernel_setmem(vecadd, 2 ,mem_C, iris_w);
    */

    iris_task task0;
    iris_task_create(&task0);
    iris_task_h2d_full(task0, mem_A, A);
    iris_task_h2d_full(task0, mem_B, B);
    void* params0[3] = { &mem_A, &mem_B, &mem_C };
    int pinfo0[3] = { iris_r, iris_r, iris_w };
    //iris_task_kernel_object(task0, vecadd, 1, NULL, &SIZE, NULL);
    iris_task_kernel(task0, "vecadd", 1, NULL, &SIZE, NULL, 3, params0, pinfo0);
    iris_task_d2h_full(task0, mem_C, C);
    iris_task_submit(task0, iris_sdq, NULL, true);
    iris_synchronize();
    auto const t_end = std::chrono::high_resolution_clock::now();
    timings.push_back(std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()));
    if(VERIFY){
      for (int i = 0; i < SIZE; i++) {
        printf("C[%d] = %d\n", i, C[i]);
        if (C[i] != (A[i] + B[i])) ERROR_COUNT++;
      }
    }
  }
  std::ofstream output_file(LOG_FILENAME);
  std::ostream_iterator<std::string> output_iterator(output_file, "\n");
  std::copy(timings.begin(), timings.end(), output_iterator);
  
  iris_finalize();
  printf("ERROR_COUNT[%d]\n", ERROR_COUNT+iris_error_count());
  return ERROR_COUNT+iris_error_count();
}
