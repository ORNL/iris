#ifndef BRISBANE_INCLUDE_BRISBANE_LLVM_H
#define BRISBANE_INCLUDE_BRISBANE_LLVM_H

#include <stdlib.h>

#if 1
#define BRISBANE_LLVM_KERNEL_BEGIN  for (size_t _z = _off2; _z < _off2 + _ndr2; _z++) \
                                    for (size_t _y = _off1; _y < _off1 + _ndr1; _y++) \
                                    for (size_t _x = _off0; _x < _off0 + _ndr0; _x++) {
#define BRISBANE_LLVM_KERNEL_END    }
#define BRISBANE_LLVM_KERNEL_ARGS   size_t _off0, size_t _off1, size_t _off2, \
                                    size_t _ndr0, size_t _ndr1, size_t _ndr2, \
                                    size_t _gws0, size_t _gws1, size_t _gws2, \
                                    size_t _lws0, size_t _lws1, size_t _lws2
#define get_global_id(N)            (N == 0 ? _x          : N == 1 ? _y         : _z)
#define get_group_id(N)             (N == 0 ? _x / _lws0  : N == 1 ? _y / _lws1 : _z / _lws2)
#define get_local_id(N)             (N == 0 ? _x % _lws0  : N == 1 ? _y & _lws1 : _z & _lws2)
#define get_global_size(N)          (N == 0 ? _gws0       : N == 1 ? _gws1      : _gws2)
#define get_local_size(N)           (N == 0 ? _lws0       : N == 1 ? _lws1      : _lws2)
#else
#define BRISBANE_LLVM_KERNEL_BEGIN  for (size_t _gz = _wgo2; _gz < _wgo2 + _wgs2; _gz++) \
                                    for (size_t _gy = _wgo1; _gy < _wgo1 + _wgs1; _gy++) \
                                    for (size_t _gx = _wgo0; _gx < _wgo0 + _wgs0; _gx++) \
                                    for (size_t _lz = 0;     _lz < _lws2; _lz++) \
                                    for (size_t _ly = 0;     _ly < _lws1; _ly++) \
                                    for (size_t _lx = 0;     _lx < _lws0; _lx++) {
#define BRISBANE_LLVM_KERNEL_END    }
#define BRISBANE_LLVM_KERNEL_ARGS   size_t _wgo0, size_t _wgo1, size_t _wgo2, \
                                    size_t _wgs0, size_t _wgs1, size_t _wgs2, \
                                    size_t _gws0, size_t _gws1, size_t _gws2, \
                                    size_t _lws0, size_t _lws1, size_t _lws2
#define get_global_id(N)            (N == 0 ? _gx * _lws0 + _lx : N == 1 ? _gy * _lws1 + _ly : _gz * _lws2 + _lz)
#define get_group_id(N)             (N == 0 ? _gx    : N == 1 ? _gy   : _gz)
#define get_local_id(N)             (N == 0 ? _lx    : N == 1 ? _ly   : _lz)
#define get_global_size(N)          (N == 0 ? _gws0  : N == 1 ? _gws1 : _gws2)
#define get_local_size(N)           (N == 0 ? _lws0  : N == 1 ? _lws1 : _lws2)
#endif

#define __kernel
#define __global                    
#define __constant
#define __local

#endif /* BRISBANE_INCLUDE_BRISBANE_LLVM_H */

