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
  std::vector<std::string> wall_timings;
  std::vector<std::string> submission_timings;
  std::vector<std::string> kernel_timings;
  std::vector<std::string> unaccounted_timings;

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

    auto const chrono_start = std::chrono::high_resolution_clock::now();
    iris_mem mem_A;
    iris_mem mem_B;
    iris_mem mem_C;
    iris_mem_create(SIZE * sizeof(int), &mem_A);
    iris_mem_create(SIZE * sizeof(int), &mem_B);
    iris_mem_create(SIZE * sizeof(int), &mem_C);

    iris_task task0;
    iris_task_create(&task0);
    iris_task_retain(task0,true);
    iris_task_h2d_full(task0, mem_A, A);
    iris_task_h2d_full(task0, mem_B, B);
    iris_task_h2d_full(task0, mem_C, C);

    void* params0[3] = { &mem_A, &mem_B, &mem_C };
    int pinfo0[3] = { iris_r, iris_r, iris_w };
    iris_task_kernel(task0, "empty", 1, NULL, &SIZE, NULL, 3, params0, pinfo0);

    iris_task_d2h_full(task0, mem_C, C);

    iris_task_submit(task0, iris_any, nullptr, true);
    iris_synchronize();
    auto const chrono_end = std::chrono::high_resolution_clock::now();
    wall_timings.push_back(std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(chrono_end - chrono_start).count()));
    size_t submission_start_time;
    size_t kernel_start_time;
    size_t kernel_end_time;
    iris_task_info(task0,iris_task_time_submit, &submission_start_time, nullptr);
    iris_task_info(task0,iris_task_time_start,  &kernel_start_time, nullptr);
    iris_task_info(task0,iris_task_time_end,    &kernel_end_time, nullptr);

    submission_timings.push_back(std::to_string(kernel_start_time - submission_start_time));
    kernel_timings.push_back(std::to_string(kernel_end_time - kernel_start_time));
    unaccounted_timings.push_back(std::to_string(
std::abs(std::chrono::duration_cast<std::chrono::nanoseconds>(chrono_end - chrono_start).count())-((kernel_start_time - submission_start_time)+(kernel_end_time - kernel_start_time))));

    iris_task_release(task0);

    if(VERIFY){
      for (int i = 0; i < SIZE; i++) {
        printf("C[%d] = %d\n", i, C[i]);
        if (C[i] != 0) ERROR_COUNT++;
      }
      printf("wall\t\t%s\n",       wall_timings.back().c_str());
      printf("submit\t\t%s\n",     submission_timings.back().c_str());
      printf("kernel\t\t%s\n",     kernel_timings.back().c_str());
      printf("unaccounted\t%s\n",  unaccounted_timings.back().c_str());
    }
  }

  std::ofstream output_file(LOG_FILENAME);
  output_file << "wall,submit,kernel,unaccounted" << std::endl;
  for(int i = 0; i < wall_timings.size(); i++){
    output_file << wall_timings[i] << "," << submission_timings[i] << "," << kernel_timings[i] << "," << unaccounted_timings[i] << std::endl;
  }
  output_file.close();
 
  iris_finalize();
  printf("ERROR_COUNT[%d]\n", ERROR_COUNT+iris_error_count());
  return ERROR_COUNT+iris_error_count();
}
