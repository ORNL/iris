#ifndef __STUB_H__
#define __STUB_H__

#ifndef ION_HEAP_ID_SYSTEM
#define ION_HEAP_ID_SYSTEM 25
#endif
#include <stdint.h>

#ifndef __hexagon__     // some defs/stubs so app can build for Hexagon simulation
#include <malloc.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

int irishxg_default_cache_flags();
int irishxg_uncached_flags();
uint8_t *irishxg_alloc_stub(int hid, int cflags, int size);
int irishxg_init_stub(int UNSIGNED_PD, int FASTRPC_QOS, int LATENCY, int DCVS_ENABLE, int hap_power_level, int use_power_level);
void irishxg_deinit_stub();
void irishxg_free_stub(void *ptr);
uint64_t irishxg_handle_stub();


#endif //__STUB_H__
