#ifndef BRISBANE_INCLUDE_BRISBANE_POLY_TYPES_H
#define BRISBANE_INCLUDE_BRISBANE_POLY_TYPES_H

#include <stddef.h>

typedef struct {
  size_t typesz;
  size_t s0;
  size_t s1;
  size_t r0;
  size_t r1;
  size_t w0;
  size_t w1;
  int dim;
} brisbane_poly_mem;

#endif /* BRISBANE_INCLUDE_BRISBANE_POLY_TYPES_H */

