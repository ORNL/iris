#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <charm/sycl.hpp>

double now() {
  static double boot = 0.0;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  if (boot == 0.0) boot = t.tv_sec + 1.e-9 * t.tv_nsec;
  return t.tv_sec + 1.e-9 * t.tv_nsec - boot;
}

int sum (int n) {
  int y = 0;
  for (int i = 0; i < n; i++)
    y += i;
  return(y);
}

void write (bool firstwrite, char* LOGFILE, float val, bool newline) {
  char LOG_BUFFER[32];

  FILE* LF_HANDLE = fopen(LOGFILE, "a");
  assert(LF_HANDLE != NULL);
  if (newline) {
    fputs("\n", LF_HANDLE);
  }
  else {
    if(firstwrite) {
      sprintf(LOG_BUFFER, "%g", val);
    }
    else {
      sprintf(LOG_BUFFER, ",%g", val);
    }
    fputs(LOG_BUFFER,LF_HANDLE);
  }
  fclose(LF_HANDLE);
}

int main(int argc, char** argv) {
  int err;
  size_t SIZE, TASK_CHAIN_LENGTH, REPEATS, nbytes;
  char* LOGFILE = NULL;
  double t0, t1;
  int* A;
  const int dims = 1;
  bool debug = false;
  bool logging = false;
  bool add_work = true; //test that the correct work is occurring on the memory buffers---when this is false, the "nothing" kernel is run which instead is designed to just measure the memory transfer

  if (argc < 3) return 1;
  SIZE = atoi(argv[1]);
  TASK_CHAIN_LENGTH = atoi(argv[2]);
  REPEATS = 1;
  if (argc == 5) {
    REPEATS = atoi(argv[3]);
    LOGFILE = argv[4];
  }
  if (argc == 6){
    REPEATS = atoi(argv[3]);
    LOGFILE = argv[4];
    add_work = argv[5];
  }
  nbytes = SIZE * sizeof(int);
  printf("[%s:%d] SIZE[%d][%luB] TASK_CHAIN_LENGTH[%d] REPEATS[%d]\n", __FILE__, __LINE__, SIZE, nbytes, TASK_CHAIN_LENGTH, REPEATS);
  sycl::queue q;

  // do for the required number of statistical samples
  for (int i = 0; i < REPEATS; ++ i) {
    // allocate and initialize the memory buffers
    A = (int*) valloc(nbytes);
    for (size_t j = 0; j < SIZE; j++) {
        A[j] = 0;
    }
    sycl::buffer<int, dims> A_dev(A, SIZE);

    if (logging) printf("Generating and issuing tasks...\n");
    t0 = now();

    // for each task in the range
    // create it with a dependency on the previous tasks memory, but assign it to the next device in the pool---thus introducing a d2d memory transfer

    int device_id = 0;
    int max_device_id;

    for (int k = 0; k < TASK_CHAIN_LENGTH; k ++) { // loop indicates the number of tasks to issue
      if (add_work) {
        //__kernel void add_id(__global int* A) {
        auto ev = q.submit([&](sycl::handler& h) {
            sycl::accessor<int, dims, sycl::access_mode::read_write> a(A_dev, h);
            h.parallel_for(sycl::range<dims>{SIZE}, [=](sycl::id<dims> const& i) {
                a[i] = a[i] + i;
            });
        });
        ev.wait();
      } else {
        //__kernel void nothing(__global int* A) {
        auto ev = q.submit([&](sycl::handler& h) {
            sycl::accessor<int, dims, sycl::access_mode::read_write> a(A_dev, h);
            h.parallel_for(sycl::range<dims>{SIZE}, [=](sycl::id<dims> const& i) {
                //
            });
        });
        ev.wait();
      }
    if (logging) printf("syncing...\n");
    //iris_synchronize();
    if (logging) printf("DONE.\n");

    t1 = now();
    //err = iris_mem_release(mem_A);
    assert(err != IRIS_ERROR && "ERROR freeing memory!");

    // verify the result
    int accelerator_sum = 0;
    for (int k = 0; k < SIZE; k++) {
      accelerator_sum += A[k];
      if (debug) printf("[%3d] %10d\n", k, A[k]);
    }
    if (debug) printf("total sum = %i, expected = %i\n", accelerator_sum, TASK_CHAIN_LENGTH*sum(SIZE));
    if (add_work) {
      assert(accelerator_sum == TASK_CHAIN_LENGTH*sum(SIZE) && "ERROR the result is incorrect!");
    }
    // log the result
    if (logging) printf("mean transfer time [%lf] ms\n", ((t1 - t0) / TASK_CHAIN_LENGTH) * 1.e+3);
    if (LOGFILE != NULL) {
      write(i == 0,LOGFILE,(t1-t0)*1.e+3,false); // save the result to file
    }
  } // end of REPEATS loop
  write(false, LOGFILE, 0.0, true); // and close the file with a new line

  return 0;
}
