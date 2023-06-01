#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <vector>

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
  iris_mem mem_A;
  int* A;
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

  iris_init(&argc, &argv, 1);
  // do for the required number of statistical samples
  for (int i = 0; i < REPEATS; ++ i) {
    // allocate and initialize the memory buffers
    A = (int*) valloc(nbytes);
    for (int j = 0; j < SIZE; j++) {
      A[j] = 0;
    }
    err = iris_mem_create(nbytes, &mem_A);
    assert(err != IRIS_ERROR && "ERROR creating memory!");

    // move the buffer to IRIS
    iris_task transfer_to;
    iris_task_create(&transfer_to);
    iris_task_h2d_full(transfer_to, mem_A, A);
    iris_task_submit(transfer_to, 1, NULL, false);

    if (logging) printf("Generating and issuing tasks...\n");
    t0 = now();

    // for each task in the range
    // create it with a dependency on the previous tasks memory, but assign it to the next device in the pool---thus introducing a d2d memory transfer

    std::vector<iris_task> tasks = { transfer_to }; // keeping track of the assigned tasks (mostly for using the last task as a dependency for new ones), but also for cleaning

    int device_id = 0;
    int max_device_id;
    err = iris_device_count(&max_device_id);
    assert(err != IRIS_ERROR && "ERROR querying platform and getting the device count!");

    for (int k = 0; k < TASK_CHAIN_LENGTH; k ++) { // loop indicates the number of tasks to issue
      if (debug) printf("Assigning task %i to device %i\n",k,device_id);
      iris_task add_id_task;
      char tname[256];
      sprintf(tname, "task_repeat_%d_tasknumber_%d", i, k);
      err = iris_task_create_name(tname, &add_id_task);
      //iris_synchronize();
      assert(err != IRIS_ERROR && "ERROR creating task!");
      size_t gws = (size_t) SIZE;
      size_t lws = (size_t) 0;
      void* params[1] = { &mem_A };
      int pinfo[1] = { iris_rw };
      if (add_work) {
        err = iris_task_kernel(add_id_task, "add_id", 1, NULL, &gws, &lws, 1, params, pinfo);
      } else {
        err = iris_task_kernel(add_id_task, "nothing", 1, NULL, &gws, &lws, 1, params, pinfo);
      }
      assert(!err && "ERROR creating memory!");
      if (!tasks.empty()) {
        err = iris_task_depend(add_id_task, 1, &tasks.back());
        assert(err != IRIS_ERROR && "ERROR adding dependency!");
      }
      err = iris_task_submit(add_id_task, device_id, NULL, false);
      assert(err != IRIS_ERROR && "ERROR submitting task!");
      tasks.push_back(add_id_task);
      device_id ++;
      if (device_id >= max_device_id) device_id = 0;
    }

    // transfer the memory buffer that's been bouncing around between the devices back to the host

    iris_task transfer_from;
    iris_task_create(&transfer_from);
    iris_task_depend(transfer_from, 1, &tasks.back());
    iris_task_d2h_full(transfer_from, mem_A, A);
    iris_task_submit(transfer_from, iris_gpu, NULL, false);

    if (logging) printf("syncing...\n");
    iris_synchronize();
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

    // free up the task queue
    //for (auto task : tasks){
    //  err = iris_task_release(task);
    //  assert(err != IRIS_ERROR && "ERROR freeing task!");
    //}

    // log the result
    if (logging) printf("mean transfer time [%lf] ms\n", ((t1 - t0) / TASK_CHAIN_LENGTH) * 1.e+3);
    if (LOGFILE != NULL) {
      write(i == 0,LOGFILE,(t1-t0)*1.e+3,false); // save the result to file
    }
  } // end of REPEATS loop
  write(false, LOGFILE, 0.0, true); // and close the file with a new line
  iris_finalize();

  return 0;
}
