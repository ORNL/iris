#ifndef IRIS_INCLUDE_IRIS_POLY_TYPES_H
#define IRIS_INCLUDE_IRIS_POLY_TYPES_H

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
} iris_poly_mem;

#endif /* IRIS_INCLUDE_IRIS_POLY_TYPES_H */

