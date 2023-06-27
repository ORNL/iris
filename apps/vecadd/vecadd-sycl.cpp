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
  std::vector<std::string> timings;
  printf("SIZE[%lu]\n", SIZE);

  sycl::queue q;
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
    auto const t_start = std::chrono::high_resolution_clock::now();
    {
      sycl::buffer<int, 1> mem_A(A, {SIZE});
      sycl::buffer<int, 1> mem_B(B, {SIZE});
      sycl::buffer<int, 1> mem_C(C, {SIZE});

      q.submit([&](sycl::handler& h) {
          sycl::accessor<int, 1, sycl::access_mode::read>  d_a(mem_A, h);
          sycl::accessor<int, 1, sycl::access_mode::read>  d_b(mem_B, h);
          sycl::accessor<int, 1, sycl::access_mode::write> d_c(mem_C, h);
          h.parallel_for(sycl::range{SIZE}, [=](sycl::id<1> i) {
              d_c[i] = d_a[i] + d_b[i];
            });
          });
      q.wait();
    }
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
  return ERROR_COUNT;
}

