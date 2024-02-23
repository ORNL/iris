#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

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

  sycl::property_list pl{sycl::property::queue::enable_profiling()};
  sycl::queue q(pl);
  printf("Running on %s\n",q.get_device().get_info<sycl::info::device::name>().c_str());

  for (int reps = 0; reps < REPEATS; reps++){
    A = (int*) valloc(SIZE * sizeof(int));
    B = (int*) valloc(SIZE * sizeof(int));
    C = (int*) valloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; i++) {
      A[i] = i;
      B[i] = i;
      C[i] = 0;
    }
    sycl::event e;
    auto const t_start = std::chrono::high_resolution_clock::now();
    {
      sycl::buffer<int, 1> mem_A(A, {SIZE});
      sycl::buffer<int, 1> mem_B(B, {SIZE});
      sycl::buffer<int, 1> mem_C(C, {SIZE});

      e = q.submit([&](sycl::handler& h) {
          sycl::accessor<int, 1, sycl::access_mode::read>  d_a(mem_A, h);
          sycl::accessor<int, 1, sycl::access_mode::read>  d_b(mem_B, h);
          sycl::accessor<int, 1, sycl::access_mode::write> d_c(mem_C, h);
          h.parallel_for(sycl::range{SIZE}, [=](sycl::id<1> i) {
            });
          });
      q.wait();
    }
    auto const t_end = std::chrono::high_resolution_clock::now();
    wall_timings.push_back(std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()));
    auto const submission_start_time = e.template get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto const kernel_start_time = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    auto const kernel_end_time = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
    submission_timings.push_back(std::to_string(kernel_start_time - submission_start_time));
    kernel_timings.push_back(std::to_string(kernel_end_time - kernel_start_time));
    unaccounted_timings.push_back(std::to_string(
        std::abs((long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() -
                 (long)((kernel_start_time - submission_start_time)+(kernel_end_time - kernel_start_time)))));
    if(VERIFY){
      for (int i = 0; i < SIZE; i++) {
        printf("C[%d] = %d\n", i, C[i]);
        if (C[i] != (A[i] + B[i])) ERROR_COUNT++;
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
  return ERROR_COUNT;
}

