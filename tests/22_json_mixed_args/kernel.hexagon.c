#include <stdio.h>
#include "AEEStdErr.h"
#include "irishxg.h"
#include <iris/iris_hexagon_imp.h>
#include "HAP_farf.h"

// profile DSP execution time (without RPC overhead) via HAP_perf api's.
#include "HAP_perf.h"
#include "HAP_compute_res.h"



#ifdef __cplusplus
extern "C" {
#endif
//case 0: iris_kernel_saxpy(saxpy_args.handle, saxpy_args.Z, saxpy_args.X, saxpy_args.Y, saxpy_args.A, saxpy_args.Xsize, saxpy_args.dspUsec, saxpy_args.dspCyc, (int)off, (int) ndr); break;
AEEResult irishxg_saxpy(
        remote_handle64 handle,
        int32 *Z,
        int Zlen,
        const int32 *X,
        int Xlen,
        const int32 *Y,
        int Ylen,
        int32 A,
        int32 *dspUsec,
        int32 *dspCyc,
        int32 off,
        int32 gws
        )
{
  int32 i = 0;
#ifdef PROFILING_ON
  uint64 startTime = HAP_perf_get_time_us();
  uint64 startCycles = HAP_perf_get_pcycles();
#endif
  for (i = off; i < off + gws; i++) {
    Z[i] = A * X[i] + Y[i];
  }

#ifdef PROFILING_ON
  uint64 endCycles = HAP_perf_get_pcycles();
  uint64 endTime = HAP_perf_get_time_us();
  *dspUsec = (int)(endTime - startTime);
  *dspCyc = (int32)(endCycles - startCycles);
  FARF(HIGH,"dilate5x5_v60 profiling over %d iterations: %d PCycles, %d microseconds. Observed clock rate %d MHz",
      LOOPS, (int)(endCycles - startCycles), (int)(endTime - startTime), 
      (int)((endCycles - startCycles) / (endTime - startTime)));
#endif
  return AEE_SUCCESS;
}
#ifdef __cplusplus
}
#endif

