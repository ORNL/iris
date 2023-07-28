#include <iris/iris.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cassert>

size_t SIZE, SIZECB;
int VERBOSE;
int REPEATS;
int DEVICES; //the number of devices to repeat the experiment over
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

  iris_timer_now(&t0);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  VERBOSE = argc > 2 ? atol(argv[2]) : 0;

  SIZECB = SIZE * SIZE * sizeof(double);

  REPEATS = 1;
  if (argc == 6) {
    DEVICES = atoi(argv[3]);
    REPEATS = atoi(argv[4]);
    LOGFILE = argv[5];
  }

  printf("SIZE[%d] MATRIX_SIZE[%lu]MB VERBOSE[%d]\n", SIZE, SIZECB / 1024 / 1024, VERBOSE);

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
    iris_timer_now(&tkern);
    for(int y = 0; y < DEVICES; y++) {
      iris_task_submit(task[y], y%num_devs, NULL, false);
      //iris_task_submit(task[y], y%num_devs, NULL, true);
    }
    
    int ret_code = iris_synchronize();
    //printf("retcode = %i\n",ret_code);
    if (ret_code == IRIS_ERROR) return 1;
    iris_timer_now(&t2);

    if (VERBOSE) {

    ijk();

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
    double gops, giops, gflops;
    gops   = 1.e-9 * ((double)op_count /(t2 - tkern));
    giops  = 1.e-9 * ((double)iop_count/(t2 - tkern));
    gflops = 1.e-9 * ((double)fop_count/(t2 - tkern));
    if (LOGFILE != NULL) {
      LF_HANDLE = fopen(LOGFILE, "a");
      assert(LF_HANDLE != NULL);
      if(i == 0){
        sprintf(LOG_BUFFER, "gops,giops,gflops\n");
      }
      sprintf(LOG_BUFFER, "%f,%f,%f\n", gops,giops,gflops);
      fputs(LOG_BUFFER,LF_HANDLE);
      fclose(LF_HANDLE);
    }
    else{
      printf("GOPs = %f, GIOPs = %f, GFLOPs = %f\n",gops,giops,gflops);
    }
  }

  return 0;
}
