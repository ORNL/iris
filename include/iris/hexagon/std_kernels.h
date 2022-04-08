/**=============================================================================

@file
   benchmark_imp.c

@brief
   implementation file for dilate filter RPC interface.

Copyright (c) 2016-2017 QUALCOMM Technologies Incorporated.
All Rights Reserved Qualcomm Proprietary
=============================================================================**/

//==============================================================================
// Include Files
//==============================================================================

// enable message outputs for profiling by defining _DEBUG and including HAP_farf.h
#ifndef _DEBUG
#define _DEBUG
#endif
#include "HAP_farf.h"

#include <string.h>

// profile DSP execution time (without RPC overhead) via HAP_perf api's.
#include "HAP_perf.h"
#include "HAP_power.h"
#include "dspCV_worker.h"

#include "AEEStdErr.h"

// includes
#include "irishxg.h"

/*===========================================================================
    DEFINITIONS
===========================================================================*/

// (128-byte is only mode supported in this example)
#define VLEN 128 
#define AHEAD 1

/*===========================================================================
    DECLARATIONS
===========================================================================*/

/*===========================================================================
    TYPEDEF
===========================================================================*/

/*===========================================================================
    LOCAL FUNCTION
===========================================================================*/

/*===========================================================================
    GLOBAL FUNCTION
===========================================================================*/

AEEResult irishxg_open(const char *uri, remote_handle64 *h) 
{
    // benchmark has no state requiring a handle. Set to a dummy value.
    *h = 0x00DEAD00;
    return 0;
}

AEEResult irishxg_close(remote_handle64 h) 
{
    return 0;
}

AEEResult irishxg_setClocks(remote_handle64 h, int32 powerLevel, int32 latency, int32 dcvsEnable, boolean useNewPowerLevel)
{

    // Set client class (useful for monitoring concurrencies)
    HAP_power_request_t request;
    memset(&request, 0, sizeof(HAP_power_request_t)); //Important to clear the structure if only selected fields are updated.
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    int retval = HAP_power_set(NULL, &request);
    if (retval) return AEE_EFAILED;


    // Configure clocks & DCVS mode
    memset(&request, 0, sizeof(HAP_power_request_t)); //Important to clear the structure if only selected fields are updated.
    request.type = HAP_power_set_DCVS_v2;

    // Implementation detail - the dcvs_enable flag actually enables some performance-boosting DCVS features
    // beyond just the voltage corners. Hence, a better way to "disable" voltage corner DCVS when not desirable
    // is to set dcvs_enable = TRUE, and instead lock min & max voltage corners to the target voltage corner. Doing this
    // trick can sometimes get better performance at minimal power cost.
    //request.dcvs_v2.dcvs_enable = dcvsEnable;   // enable dcvs if desired, else it locks to target corner
    request.dcvs_v2.dcvs_enable = TRUE;
    request.dcvs_v2.dcvs_params.target_corner = powerLevel;

    if (FALSE == useNewPowerLevel) {
      // convert benchmark application power levels to dcvs_v2 clock levels.
      FARF(HIGH,"using old power levels");
      const uint32_t numPowerLevels = 6;
      const HAP_dcvs_voltage_corner_t voltageCorner[numPowerLevels]
          = { HAP_DCVS_VCORNER_TURBO,
              HAP_DCVS_VCORNER_NOMPLUS,
              HAP_DCVS_VCORNER_NOM,
              HAP_DCVS_VCORNER_SVSPLUS,
              HAP_DCVS_VCORNER_SVS,
              HAP_DCVS_VCORNER_SVS2 };

      if ((uint32_t)powerLevel >= numPowerLevels) powerLevel = numPowerLevels - 1;
      request.dcvs_v2.dcvs_params.target_corner = voltageCorner[powerLevel];
    }


    if (dcvsEnable)
    {
        request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE; // no minimum
        request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE; // no maximum
    }
    else
    {
        request.dcvs_v2.dcvs_params.min_corner = request.dcvs_v2.dcvs_params.target_corner;  // min corner = target corner
        request.dcvs_v2.dcvs_params.max_corner = request.dcvs_v2.dcvs_params.target_corner;  // max corner = target corner
    }
    
    request.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.set_latency = TRUE;
    request.dcvs_v2.latency = latency;
    retval = HAP_power_set(NULL, &request);
    if (retval) return AEE_EFAILED;
    
// vote for HVX power
    memset(&request, 0, sizeof(HAP_power_request_t)); //Important to clear the structure if only selected fields are updated.
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    retval = HAP_power_set(NULL, &request);
    if (retval) return AEE_EFAILED;

    return AEE_SUCCESS;
}


