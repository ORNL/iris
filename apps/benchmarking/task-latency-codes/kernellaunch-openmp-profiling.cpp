#include "../timer.h"
#include "../utils.h"
#include "../kernel.openmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double t0, t1;
double i0, i1;
double f0, f1;
double c0, c1;

float* dX;
float* dY;
float* dZ;

int SIZE;
int LOOP;
int REPEATS;
size_t nbytes;
float A;

char* LOGFILE = NULL;
FILE* LF_HANDLE;
char LOG_BUFFER[32];

void Init(){
  dX = (float*) malloc(nbytes);
  dY = (float*) malloc(nbytes);
  dZ = (float*) malloc(nbytes);
}

void Compute(int loop) {
  for (int i = 0; i < loop; i++)  {
    saxpy(dZ, A, dX, dY, 0, SIZE);
  }
}

void Finalize(){
  free(dX);
  free(dY);
  free(dZ);
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  SIZE = atoi(argv[1]);
  LOOP = atoi(argv[2]);
  nbytes = SIZE * sizeof(float);

  REPEATS = 1;
  if (argc == 5) {
    REPEATS = atoi(argv[3]);
    LOGFILE = argv[4];
  }
  printf("[%s:%d] SIZE[%d][%luB] LOOP[%d] REPEATS[%d]\n", __FILE__, __LINE__, SIZE, nbytes, LOOP,REPEATS);
  t0 = now();

  i0 = now();
  Init();
  i1 = now();

  for (int i = 0; i < REPEATS; ++ i) {
    c0 = now();
    Compute(LOOP);
    c1 = now();

    printf("latency [%lf] ms\n", ((c1 - c0) / LOOP) * 1.e+6);
    if (LOGFILE != NULL) {
        LF_HANDLE = fopen(LOGFILE, "a");
        assert(LF_HANDLE != NULL);
        if(i == 0){
          sprintf(LOG_BUFFER, "%g", (c1-c0)*1.e+6);
        }
        else {
          sprintf(LOG_BUFFER, ",%g", (c1-c0)*1.e+6);
        }
        fputs(LOG_BUFFER,LF_HANDLE);
        fclose(LF_HANDLE);
      }
  }
  if (LOGFILE != NULL) {
      LF_HANDLE = fopen(LOGFILE, "a");
      assert(LF_HANDLE != NULL);
      fputs("\n", LF_HANDLE);
      fclose(LF_HANDLE);
  }

  f0 = now();
  Finalize();
  f1 = now();

  t1 = now();

  printf("secs: T[%lf] I[%lf] C[%lf] F[%lf]\n", t1 - t0, i1 - i0, c1 - c0, f1 - f0);

  return 0;
}

