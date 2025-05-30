#include <iris/iris.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include "iris/verify.h"
#include "iris/gettime.h"
#ifdef SNAPDRAGON
#include "iris/hexagon/stub.h"
#endif
#include "iris/iris_macros.h"
#include "saxpy.iris.h"

typedef int8_t boolean;
#define TRUE 1
#define FALSE 0
#define MIN(X, Y) ((X)<(Y) ? (X) : (Y))
extern "C" void saxpy(int32_t *Z,
               int32_t *X,
               int32_t *Y,
               int32_t size, 
               int32_t A, 
               size_t, size_t);
void saxpy_ref(int32_t *Z,
               int32_t *X,
               int32_t *Y,
               int32_t A, 
               int32_t size);
//#define DISABLE_IRIS
int main(int argc, char** argv) {
#ifndef DISABLE_IRIS
  iris_init(&argc, &argv, 1);
#endif

  int SIZE;
  int TARGET, target_dev;
  int MUL;
  int i, nErr;
  int32_t *X, *Y, *Z, *Zref;
  int32_t A = 10;
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

  SIZE = argc > 1 ? atol(argv[1]) : 8;
  TARGET = argc > 2 ? atol(argv[2]) : 0;
  MUL = argc > 3 ? atol(argv[3]) : 1;
  char *arch = getenv("ARCH");
  if (arch != NULL && argc<=2)
      TARGET = atoi(arch);
  target_dev = TARGET == 0 ? iris_cpu : TARGET == 1 ? iris_gpu : TARGET == 2 ? iris_dsp : iris_fpga;

  printf("[%s:%d] SIZE[%d] TARGET[%d][0,1,2][cpu,gpu,dsp] MUL[%d]\n", __FILE__, __LINE__, SIZE, TARGET, MUL);

  X = (int32_t*) malloc(SIZE * sizeof(int32_t));
  Y = (int32_t*) malloc(SIZE * sizeof(int32_t));
  Z = (int32_t*) malloc(SIZE * sizeof(int32_t));
  Zref = (int32_t*) malloc(SIZE * sizeof(int32_t));

  for (i = 0; i < SIZE; i++) {
    X[i] = i+1;
    Y[i] = i+1;
  }

  printf("X [");
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", X[i]);
  printf("]\n");
  printf("Y [");
  int32_t cuUsec = 0, cuCyc = 0;
  int32_t *cuUsecPtr = (int32_t*)&cuUsec;
  int32_t *cuCycPtr = (int32_t*)&cuCyc;
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %2d", Y[i]);
  printf("]\n");
#ifdef ENABLE_HEXAGON
  if (target_dev == iris_dsp) {
      irishxg_init_stub(UNSIGNED_PD, FASTRPC_QOS, LATENCY, DCVS_ENABLE, hap_power_level, use_power_level);
  }
#endif
  //printf("Enabled shared memory model\n");
  unsigned long long t1 = GetTime();
#ifndef DISABLE_IRIS
  if (TARGET / 10 > 0)
      iris_set_shared_memory_model(1);
#endif
//#define ORIGINAL
//#define ALL_APIS
#ifdef DISABLE_IRIS
  saxpy(Z, X, Y, SIZE, A, (size_t)0, (size_t)0); // CPU, DSP
#else
  //printf("X:%p Y:%p Z:%p\n", X, Y, Z);
#if 0
  IRIS_SINGLE_TASK(task0, "saxpy", target_dev, 1,
          NULL_OFFSET, GWS(SIZE), NULL_LWS,
          OUT_TASK(Z, int32_t *, int32_t, Z, sizeof(int32_t)*SIZE),
          IN_TASK(X, const int32_t *, int32_t, X, sizeof(int32_t)*SIZE),
          IN_TASK(Y, const int32_t *, int32_t, Y, sizeof(int32_t)*SIZE),
          PARAM(SIZE, int32_t, iris_cpu),
          PARAM(A, int32_t),
          PARAM(cuUsecPtr, int32_t*, iris_dsp),
          PARAM(cuCycPtr, int32_t*, iris_dsp));
#else
  isaxpy_cpp(target_dev, Z, X, Y, SIZE, A, cuUsecPtr, cuCycPtr);
#endif
#endif
  unsigned long long t2 = GetTime();
  unsigned int rpcOverhead = (unsigned int) (t2 - t1 - cuUsec);
  printf("total time %d uSec, DSP-measured %d uSec and %d cyc (IRIS+RPC overhead %d uSec), observed clock %d MHz\n",
                  (int) (t2-t1), cuUsec, cuCyc, rpcOverhead, (cuUsec == 0 ? 0 : cuCyc/cuUsec));
  
  printf("S = %d * X + Y [", A);
  for (i = 0; i < MIN(8, SIZE); i++) printf(" %3d", Z[i]);
  printf("]\n");

  saxpy_ref(Zref, X, Y, A, SIZE);
  int bitexactErrors = 0;
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
  VERIFY(0 == bitexactErrors);

  free(X);
  free(Y);
  free(Z);


#ifndef DISABLE_IRIS
  iris_finalize();
#endif
  bail:

  return 0;
}
