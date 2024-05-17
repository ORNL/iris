#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cassert>
#include <limits.h>

bool DEBUG=false;
size_t SIZE, SIZECB;
int VERIFY;
int REPEATS;
int DEVICES; //the number of concurrent tasks to repeat the experiment over
char* LOGFILE = NULL;
FILE* LF_HANDLE;
char LOG_BUFFER[32];

double *A, *B, *C;
double t0, t1, t2, t3, tkern;

void ijk() {
#pragma iris kernel h2d(A[0:SIZE*SIZE], B[0:SIZE*SIZE]) d2h(C[0:SIZE*SIZE]) device(all)
#pragma iris data access index(i) h2d(A[i*SIZE:SIZE], B[0:SIZE*SIZE]) d2h(C[i*SIZE:SIZE])
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      double sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] = sum;
    }
  }
}

void kij() {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      double a = A[i * SIZE + k];
      for (int j = 0; j < SIZE; j++) {
        C[i * SIZE + j] += a * B[k * SIZE + j];
      }
    }
  }
}

int main(int argc, char** argv) {
  int ERROR = 0;

  iris_init(&argc, &argv, true);
  iris_overview();

  iris_timer_now(&t0);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  VERIFY = argc > 2 ? atol(argv[2]) : 0;

  SIZECB = SIZE * SIZE * sizeof(double);

  REPEATS = 1;
  if (argc == 6) {
    DEVICES = atoi(argv[3]);
    REPEATS = atoi(argv[4]);
    LOGFILE = argv[5];
  }

  printf("SIZE[%lu] MATRIX_SIZE[%lu]MB VERIFY[%d]\n", SIZE, SIZECB / 1024 / 1024, VERIFY);

  //set the repeats argument for a good statistical sample
  for (int z = 0; z < REPEATS; ++ z) {
    iris_mem mem_A[DEVICES];
    iris_mem mem_B[DEVICES];
    iris_mem mem_C[DEVICES];
    iris_task task[DEVICES];

    A = (double*) malloc(SIZE * SIZE * sizeof(double));
    B = (double*) malloc(SIZE * SIZE * sizeof(double));
    C = (double*) malloc(SIZE * SIZE * sizeof(double));

    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++) {
        A[i * SIZE + j] = i + j;
        B[i * SIZE + j] = i * j;
        C[i * SIZE + j] = 0.0;
      }
    }

    iris_timer_now(&t1);
    for(int y = 0; y < DEVICES; y++) {
      iris_mem_create(SIZE * SIZE * sizeof(double), &mem_A[y]);
      iris_mem_create(SIZE * SIZE * sizeof(double), &mem_B[y]);
      iris_mem_create(SIZE * SIZE * sizeof(double), &mem_C[y]);

      iris_task_create(&task[y]);
      iris_task_retain(task[y],true);
      //iris_task_create_name("stub",&task[y]);
      iris_task_h2d_full(task[y], mem_A[y], A);
      iris_task_h2d_full(task[y], mem_B[y], B);
      size_t ijk_idx[2] = { SIZE, SIZE };
      //size_t ijk_lws[2] = { 32, 32 };
      void* params[3] = { &mem_C[y], &mem_A[y], &mem_B[y] };
      int pinfo[3] = { iris_w, iris_r, iris_r };
      //iris_task_kernel(task[y], "ijk", 2, NULL, ijk_idx, ijk_lws, 3, params, pinfo);
      iris_task_kernel(task[y], "ijk", 2, NULL, ijk_idx, NULL, 3, params, pinfo);
      iris_task_d2h_full(task[y], mem_C[y], C);
    }
    
    int num_devs;
    iris_device_count(&num_devs);
    for(int y = 0; y < num_devs; y++) {
      size_t _type, size;
      iris_device_info(y, iris_type, &_type, &size);
      assert(_type != iris_cpu);
      //printf("type = %i cpu = %i\n",_type,iris_cpu);
    }
    iris_timer_now(&tkern);
    for(int y = 0; y < DEVICES; y++) {
      iris_task_submit(task[y], y%num_devs, NULL, false);
      //iris_task_submit(task[y], y%num_devs, NULL, true);
    }
    
    int ret_code = iris_synchronize();
    if (ret_code == IRIS_ERROR) return 1;
    iris_timer_now(&t2);

    size_t submission_start_time;
    size_t kernel_start_time;
    size_t kernel_end_time;
    size_t rolling_time = 0;
    size_t earliest_kernel_time = UINT_MAX;
    size_t last_kernel_completion_time = 0;

    for(int y = 0; y < DEVICES; y++){
      iris_task_info(task[y],iris_task_time_submit, &submission_start_time, nullptr);
      iris_task_info(task[y],iris_task_time_start,  &kernel_start_time, nullptr);
      iris_task_info(task[y],iris_task_time_end,    &kernel_end_time, nullptr);
      iris_task_release(task[y]);
      earliest_kernel_time = std::min(earliest_kernel_time,kernel_start_time);
      last_kernel_completion_time = std::max(last_kernel_completion_time,kernel_end_time);
      rolling_time += (kernel_end_time - kernel_start_time);
    }
    float event_elapsed_time = (last_kernel_completion_time-earliest_kernel_time)*1.e-9;
    printf("rolling_time (s) = %f, event kernel time (s) = %f, iris timers (s) = %f, kernel time (s) = %f\n",rolling_time*1.e-9, event_elapsed_time, t2-t1, t2-tkern);
    if (DEBUG | VERIFY){
      ijk();
    }

    if (DEBUG) {
    printf("[[ A ]]\n");
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++) {
        printf("%5.0lf ", A[i * SIZE + j]);
      }
      printf("\n");
    }

    printf("[[ B ]]\n");
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++) {
        printf("%5.0lf ", B[i * SIZE + j]);
      }
      printf("\n");
    }

    printf("[[ C ]]\n");
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++) {
        printf("%5.0lf ", C[i * SIZE + j]);
      }
      printf("\n");
    }
    }
    if(VERIFY){
    printf("Checking errors\n");
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++) {
        double sum = 0.0;
        for (int k = 0; k < SIZE; k++) {
          sum += A[i * SIZE + k] * B[k * SIZE + j];
        }
        if (sum != C[i * SIZE + j]) ERROR++;
      }
    }
    }
    iris_timer_now(&t3);
    printf("ERROR[%d] TIME[%lf,%lf]\n", ERROR, t3 - t0, t2 - t1);

    for(int y = 0; y < DEVICES; y++){
      //iris_task_release(task[y]);
      iris_mem_release(mem_A[y]);
      iris_mem_release(mem_B[y]);
      iris_mem_release(mem_C[y]);
    }
    free(A);
    free(B);
    free(C);

    iris_finalize();
    long i=SIZE; long j=SIZE; long k=SIZE;
    long long op_count, iop_count, fop_count;
    op_count = i*j*k*6*DEVICES;
    iop_count= i*j*k*4*DEVICES;
    fop_count= i*j*k*2*DEVICES;
    printf("Total FLOP count = %llu, GFLOPS based on event kernel time (s) = %f, GFLOPS Based on host kernel time (s) = %f\n",fop_count,(double)(fop_count/event_elapsed_time)*1.e-9, fop_count/(t2-tkern)*1.e-9);

    double gops, giops, gflops, tflops;
    gops   = 1.e-9 * ((double)op_count /(t2 - tkern));
    giops  = 1.e-9 * ((double)iop_count/(t2 - tkern));
    gflops = 1.e-9 * ((double)fop_count/(t2 - tkern));
    //gflops = (double)(fop_count/event_elapsed_time)*1.e-9;
    tflops = (double)(fop_count/event_elapsed_time)*1.e-12;

    if (LOGFILE != NULL) {
      LF_HANDLE = fopen(LOGFILE, "a");
      assert(LF_HANDLE != NULL);
      if(i == 0){
        sprintf(LOG_BUFFER, "gops,giops,gflops,tflops\n");
      }
      sprintf(LOG_BUFFER, "%f,%f,%f,%f\n", gops,giops,gflops,tflops);
      fputs(LOG_BUFFER,LF_HANDLE);
      fclose(LF_HANDLE);
    }
    else{
      printf("GOPs = %f, GIOPs = %f, GFLOPs = %f, TFLOPS= %f\n",gops,giops,gflops,tflops);
    }
  }

  return 0;
}
