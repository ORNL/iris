#include "AEEStdErr.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "iris/verify.h"
#include "irishxg.h"
#include "remote.h"
#include "rpcmem.h" // helper API's for shared buffer allocation
#include "stub.h"

#pragma weak remote_session_control  

remote_handle64 __handle;
uint64_t irishxg_handle_stub()
{
    return __handle;
}
int irishxg_default_cache_flags()
{
    return RPCMEM_DEFAULT_FLAGS;
}
int irishxg_uncached_flags()
{
    return RPCMEM_FLAG_UNCACHED;
}
uint8_t *irishxg_alloc_stub(int hid, int cflags, int size)
{
    //printf("alloc_stub is being called now size:%d\n", size);
    //if (hid == -1)
    hid = ION_HEAP_ID_SYSTEM;
    if (cflags == -1)
        cflags = RPCMEM_DEFAULT_FLAGS;
    return (uint8_t*)rpcmem_alloc(hid, cflags, size);
}
int irishxg_init_stub(int UNSIGNED_PD, int FASTRPC_QOS, int LATENCY, int DCVS_ENABLE, int hap_power_level, int use_power_level)
{
    int nErr = 0;
    int retVal = 0;
    rpcmem_init();
    // Unsigned PD
    if (1 == UNSIGNED_PD)
    {
        if (remote_session_control)
        {
            struct remote_rpc_control_unsigned_module data;
            data.enable = 1;
            data.domain = CDSP_DOMAIN_ID;
            retVal = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data));
            printf("remote_session_control returned %d for configuring unsigned PD.\n", retVal);
        }
        else
        {
            printf("Unsigned PD not supported on this device.\n");
        }
    }
#ifndef DSP_TAG
#define DSP_TAG cdsp
#endif
#define DSP_TAG_STR(X) #X
#define DSP_STR(X) DSP_TAG_STR(X)
    // Open a handle on benchmark RPC interface. Choose default URI for compute DSP. 
    // See remote.idl for more options/information.
    __handle = -1;
    char *irishxg_URI_Domain = irishxg_URI "&_dom=" DSP_STR(DSP_TAG);		// try opening handle on CDSP.
    retVal = irishxg_open(irishxg_URI_Domain, &__handle);
    if (retVal)
        printf("Error 0x%x: unable to create fastrpc session on CDSP\n", retVal);
    if (retVal == 0) {
        if (1 == FASTRPC_QOS)
        {
            struct remote_rpc_control_latency data;
            data.enable = 1;
            remote_handle64_control(__handle, DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
        }

        printf("setting clocks to power level %d, Deprecated power level NOT used %d\n", hap_power_level, use_power_level);
        retVal = irishxg_setClocks(__handle, hap_power_level, LATENCY, DCVS_ENABLE, use_power_level);
        VERIFY(0 == retVal);
    }
bail:
    return retVal;
}
void irishxg_deinit_stub()
{
    rpcmem_deinit();
}
void irishxg_free_stub(void *ptr)
{
    rpcmem_free(ptr);
}
