#include <iris/iris.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include "iris/verify.h"
#include "iris/gettime.h"
#include "iris/iris_macros.h"

void saxpy_ref(double *Z,
               double *X,
               double *Y,
               double A, 
               int32_t size)
{
    int i;
    for (i=0; i<size; i++) {
        Z[i] = A * X[i] + Y[i];
    }
}

typedef int8_t boolean;
#define TRUE 1
#define FALSE 0
#define MIN(X, Y) ((X)<(Y) ? (X) : (Y))
int main(int argc, char** argv) {
  iris_init(&argc, &argv, 1);

  size_t SIZE, SIZECB;
  int TARGET;
  int MUL;
  int i;
  double *X, *Y, *Z, *Zref;
  double A = 10;
  int bitexactErrors = 0;
  SIZE = argc > 1 ? atol(argv[1]) : 8;
  SIZECB = SIZE * sizeof(double);
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  MUL = argc > 3 ? atol(argv[3]) : 1;
  //char *arch = getenv("ARCH");
  //if (arch != NULL && argc<=2)
  //    TARGET = atoi(arch);
  //target_dev = TARGET == 0 ? iris_cpu : TARGET == 1 ? iris_gpu : TARGET == 2 ? iris_dsp : iris_fpga;
  int target1 = iris_roundrobin;
  int target2 = iris_depend;

  printf("[%s:%d] SIZE[%zu] TARGET[%d][0,1,2][cpu,gpu,dsp] MUL[%d]\n", __FILE__, __LINE__, SIZE, TARGET, MUL);
  //record static option
  {
  iris_record_start();
  X = (double*) malloc(SIZECB);
  Y = (double*) malloc(SIZECB);
  Z = (double*) malloc(SIZECB);
  Zref = (double*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = (double)i+1;
    Y[i] = (double)i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);

  void* json_inputs[11] = { &SIZE, &SIZECB, Z, X, Y, &memZ, &memX, &memY, &A, &target1, &target2 };

  iris_graph graph;
  iris_graph_create_json("graph_data_size.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", Y[i]);
  printf("]\n");

  printf("S = %f * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3f", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%f, dst=%f\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);

  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the record static (data_size) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  iris_record_stop();
  printf("Passed the record static (data_size) test!\n");
  }

  //replay static option
  {
  X = (double*) malloc(SIZECB);
  Y = (double*) malloc(SIZECB);
  Z = (double*) malloc(SIZECB);
  Zref = (double*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = (double) i+1;
    Y[i] = (double) i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);

  void* json_inputs[6] = { Z, X, Y, &memZ, &memX, &memY};

  iris_graph graph;
  iris_graph_create_json("output.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", Y[i]);
  printf("]\n");

  printf("S = %f * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3f", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%f, dst=%f\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);

  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the record static (data_size) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the replay static (data_size) test!\n");
  }

  //static record option (data_type)
  {
  X = (double*) malloc(SIZECB);
  Y = (double*) malloc(SIZECB);
  Z = (double*) malloc(SIZECB);
  Zref = (double*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = (double) i+1;
    Y[i] = (double) i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);

  void* json_inputs[11] = { &SIZE, &SIZECB, Z, X, Y, &memZ, &memX, &memY, &A, &target1, &target2 };

  iris_graph graph;
  iris_graph_create_json("graph_data_type.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", Y[i]);
  printf("]\n");

  printf("S = %f * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3f", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%f, dst=%f\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);
  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the record static (data_type) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the record static (data_type) test!\n");
  }

  //replay static (data_type) option
  {
  X = (double*) malloc(SIZECB);
  Y = (double*) malloc(SIZECB);
  Z = (double*) malloc(SIZECB);
  Zref = (double*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = (double) i+1;
    Y[i] = (double) i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);

  void* json_inputs[6] = { Z, X, Y, &memZ, &memX, &memY};

  iris_graph graph;
  iris_graph_create_json("output.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", Y[i]);
  printf("]\n");

  printf("S = %f * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3f", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%f, dst=%f\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);

  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the record static (data_type) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the replay static (data_type) test!\n");
  }

  //dynamic record option
  {
  X = (double*) malloc(SIZECB);
  Y = (double*) malloc(SIZECB);
  Z = (double*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = (double) i+1;
    Y[i] = (double) i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);
  int data_size = sizeof(double);
  void* json_inputs[12] = { &SIZE, &SIZECB, Z, X, Y, &memZ, &memX, &memY, &A, &data_size, &target1, &target2 };

  iris_graph graph;
  iris_graph_create_json("graph_dynamic_data_size.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", Y[i]);
  printf("]\n");

  printf("S = %f * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3f", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%f, dst=%f\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);
  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the record dynamic (data_size) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the record dynamic (data_size) test!\n");
  }
  //replay dynamic (data_size) option
  {
  X = (double*) malloc(SIZECB);
  Y = (double*) malloc(SIZECB);
  Z = (double*) malloc(SIZECB);
  Zref = (double*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = (double) i+1;
    Y[i] = (double) i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);

  void* json_inputs[6] = { Z, X, Y, &memZ, &memX, &memY};

  iris_graph graph;
  iris_graph_create_json("output.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2f", Y[i]);
  printf("]\n");

  printf("S = %f * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3f", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%f, dst=%f\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);

  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the record dynamic (data_size) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the replay dynamic (data_size) test!\n");
  }

  iris_finalize();

  return bitexactErrors+iris_error_count();
}
