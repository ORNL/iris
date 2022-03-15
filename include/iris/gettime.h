#ifndef __IRIS_GET_TIME_H__
#define __IRIS_GET_TIME_H__
#ifdef __hexagon__     // some defs/stubs so app can build for Hexagon simulation
#include "hexagon_sim_timer.h"
#include "hexagon_cache.h" // for removing buffers from cache during simulation/profiling
#define GetTime hexagon_sim_read_pcycles        // For Hexagon sim, use PCycles for profiling
#else
#include <stdlib.h>
#include <sys/time.h>
static unsigned long long GetTime( void )
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

#endif
#endif // __IRIS_GET_TIME_H__
