#include <iris/iris.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include "iris/verify.h"
#include "iris/gettime.h"
#include "iris/iris_macros.h"
#include <signal.h>
void saxpy_ref(int32_t *Z,
               int32_t *X,
               int32_t *Y,
               int32_t A, 
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
  int TARGET, target_dev;
  int MUL;
  int i, nErr;
  int32_t *X, *Y, *Z, *Zref;
  int A = 10;
  boolean use_power_level = TRUE;             //default: Use the new power level
  boolean power_level_dep_cmd = FALSE;        //default: Flag to indicate if deprecated power level option is sent as command arg
  boolean power_level_cmd = FALSE;            //default: Flag to indicate if new power level option is sent as command arg
  int POWER_LEVEL = 6;                        //Set the default value to TURBO which will work in all targets.
  int POWER_LEVEL_DEP = 0;                    //Set the default value of the deprecated to the highest performance.
  int hap_power_level = POWER_LEVEL;          //Set the default value to the highest performance.
  use_power_level = (power_level_dep_cmd == FALSE) ? TRUE : (power_level_cmd == TRUE) ? TRUE : FALSE;
  hap_power_level = (use_power_level == TRUE) ? POWER_LEVEL : POWER_LEVEL_DEP;
  int DCVS_ENABLE = 0;
  int LATENCY = 100;
  int FASTRPC_QOS = 0;
  int UNSIGNED_PD = 0;
  int COMPUTE_RES = 0;
  int bitexactErrors = 0;
  SIZE = argc > 1 ? atol(argv[1]) : 8;
  SIZECB = SIZE * sizeof(int);
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  MUL = argc > 3 ? atol(argv[3]) : 1;
  //char *arch = getenv("ARCH");
  //if (arch != NULL && argc<=2)
  //    TARGET = atoi(arch);
  //target_dev = TARGET == 0 ? iris_cpu : TARGET == 1 ? iris_gpu : TARGET == 2 ? iris_dsp : iris_fpga;
  int target1 = iris_roundrobin;
  int target2 = iris_depend;

  printf("[%s:%d] SIZE[%zu] TARGET[%d][0,1,2][cpu,gpu,dsp] MUL[%d]\n", __FILE__, __LINE__, SIZE, TARGET, MUL);
  //static option
  {
  X = (int32_t*) malloc(SIZECB);
  Y = (int32_t*) malloc(SIZECB);
  Z = (int32_t*) malloc(SIZECB);
  Zref = (int32_t*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = i+1;
    Y[i] = i+1;
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
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", Y[i]);
  printf("]\n");

  printf("S = %d * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3d", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%d, dst=%d\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);
  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the static (data_size) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the static (data_size) test!\n");
  }

  //static option (data_type)
  {
  X = (int32_t*) malloc(SIZECB);
  Y = (int32_t*) malloc(SIZECB);
  Z = (int32_t*) malloc(SIZECB);
  Zref = (int32_t*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = i+1;
    Y[i] = i+1;
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
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", Y[i]);
  printf("]\n");

  printf("S = %d * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3d", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%d, dst=%d\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);
  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the static (data_type) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the static (data_type) test!\n");
  }

  //dynamic option
  {
  X = (int32_t*) malloc(SIZECB);
  Y = (int32_t*) malloc(SIZECB);
  Z = (int32_t*) malloc(SIZECB);

  for (i = 0; i < SIZE; i++) {
    X[i] = i+1;
    Y[i] = i+1;
  }

  iris_mem memX;
  iris_mem memY;
  iris_mem memZ;
  iris_mem_create(SIZECB, &memX);
  iris_mem_create(SIZECB, &memY);
  iris_mem_create(SIZECB, &memZ);
  int data_size = sizeof(int);
  void* json_inputs[12] = { &SIZE, &SIZECB, Z, X, Y, &memZ, &memX, &memY, &A, &data_size, &target1, &target2 };

  iris_graph graph;
  iris_graph_create_json("graph_dynamic_data_size.json", json_inputs, &graph);

  iris_graph_submit(graph, iris_sdq, 1);
  iris_synchronize();
 
  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", X[i]);
  printf("]\n");
  printf("Y [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", Y[i]);
  printf("]\n");

  printf("S = %d * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3d", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  printf( "Checking for bit-exact errors... \n");
  for (i=0; i<SIZE; i++)
  {
      if (Zref[i] != Z[i]) {
          bitexactErrors++;
          if (bitexactErrors < 10 )
              printf("Bit exact error: i=%d, refval=%d, dst=%d\n",i, Zref[i], Z[i]);
      }
  }
  printf( "Number of bit-exact errors: %d \n", bitexactErrors);

  free(X);
  free(Y);
  free(Z);
  if (bitexactErrors != 0){
    iris_finalize();
    printf("Failed the dynamic (data_size) test!\n");
    return(bitexactErrors+iris_error_count());
  }
  printf("Passed the dynamic (data_size) test!\n");
  }  //TODO: fix/check this test with different data_size
  //TODO: fix/check this test with setting the data_size from input resolve
  //TODO: fix/check this test with different data_type
  
  iris_finalize();

  return bitexactErrors+iris_error_count();
}
