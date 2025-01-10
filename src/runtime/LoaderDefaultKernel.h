#ifndef IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H
#define IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H

#include "Loader.h"

#include "HostInterface.h"

namespace iris {
namespace rt {

class LoaderDefaultKernel : public HostInterfaceClass {
public:
  LoaderDefaultKernel(const char *path);
  ~LoaderDefaultKernel();

  int LoadFunctions();
#define RESET_DECL_SEQ(IT, T, M)  void (*iris_reset_ ## M)(T *arr, T value, size_t size, void *stream) ;
  RESET_DECL_SEQ(iris_uint8,  uint8_t,  u8);
  RESET_DECL_SEQ(iris_uint16, uint16_t, u16);
  RESET_DECL_SEQ(iris_uint32, uint32_t, u32);
  RESET_DECL_SEQ(iris_uint64, uint64_t, u64);
  RESET_DECL_SEQ(iris_int8,   int8_t,   i8);
  RESET_DECL_SEQ(iris_int16,  int16_t,  i16);
  RESET_DECL_SEQ(iris_int32,  int32_t,  i32);
  RESET_DECL_SEQ(iris_int64,  int64_t,  i64);
  RESET_DECL_SEQ(iris_float,  float,    f32);
  RESET_DECL_SEQ(iris_double, double,   f64);

#define ARITH_DECL_SEQ(IT, T, M)  void (*iris_arithmetic_seq_ ## M)(T *arr, T start, T step, size_t size, void *stream) ;
  ARITH_DECL_SEQ(iris_uint8,  uint8_t,  u8);
  ARITH_DECL_SEQ(iris_uint16, uint16_t, u16);
  ARITH_DECL_SEQ(iris_uint32, uint32_t, u32);
  ARITH_DECL_SEQ(iris_uint64, uint64_t, u64);
  ARITH_DECL_SEQ(iris_int8,   int8_t,   i8);
  ARITH_DECL_SEQ(iris_int16,  int16_t,  i16);
  ARITH_DECL_SEQ(iris_int32,  int32_t,  i32);
  ARITH_DECL_SEQ(iris_int64,  int64_t,  i64);
  ARITH_DECL_SEQ(iris_float,  float,    f32);
  ARITH_DECL_SEQ(iris_double, double,   f64);

#define GEOM_DECL_SEQ(IT, T, M)  void (*iris_geometric_seq_ ## M)(T *arr, T start, T step, size_t size, void *stream) ;
  GEOM_DECL_SEQ(iris_uint8,  uint8_t,  u8);
  GEOM_DECL_SEQ(iris_uint16, uint16_t, u16);
  GEOM_DECL_SEQ(iris_uint32, uint32_t, u32);
  GEOM_DECL_SEQ(iris_uint64, uint64_t, u64);
  GEOM_DECL_SEQ(iris_int8,   int8_t,   i8);
  GEOM_DECL_SEQ(iris_int16,  int16_t,  i16);
  GEOM_DECL_SEQ(iris_int32,  int32_t,  i32);
  GEOM_DECL_SEQ(iris_int64,  int64_t,  i64);
  GEOM_DECL_SEQ(iris_float,  float,    f32);
  GEOM_DECL_SEQ(iris_double, double,   f64);
#define RANDOM_DECL_SEQ(IT, RTYPE, T, TAG)  void (*iris_random_ ## RTYPE ## _seq_ ## TAG)(T *arr, unsigned long long seed, T p1, T p2, size_t size, void *stream);
  RANDOM_DECL_SEQ(iris_uint8,  uniform, uint8_t,  u8);
  RANDOM_DECL_SEQ(iris_uint16, uniform, uint16_t, u16);
  RANDOM_DECL_SEQ(iris_uint32, uniform, uint32_t, u32);
  RANDOM_DECL_SEQ(iris_uint64, uniform, uint64_t, u64);
  RANDOM_DECL_SEQ(iris_int8,   uniform, int8_t,   i8);
  RANDOM_DECL_SEQ(iris_int16,  uniform, int16_t,  i16);
  RANDOM_DECL_SEQ(iris_int32,  uniform, int32_t,  i32);
  RANDOM_DECL_SEQ(iris_int64,  uniform, int64_t,  i64);
  RANDOM_DECL_SEQ(iris_float,  uniform, float,    f32);
  RANDOM_DECL_SEQ(iris_double, uniform, double,   f64);

  RANDOM_DECL_SEQ(iris_uint8,  normal, uint8_t,  u8);
  RANDOM_DECL_SEQ(iris_uint16, normal, uint16_t, u16);
  RANDOM_DECL_SEQ(iris_uint32, normal, uint32_t, u32);
  RANDOM_DECL_SEQ(iris_uint64, normal, uint64_t, u64);
  RANDOM_DECL_SEQ(iris_int8,   normal, int8_t,   i8);
  RANDOM_DECL_SEQ(iris_int16,  normal, int16_t,  i16);
  RANDOM_DECL_SEQ(iris_int32,  normal, int32_t,  i32);
  RANDOM_DECL_SEQ(iris_int64,  normal, int64_t,  i64);
  RANDOM_DECL_SEQ(iris_float,  normal, float,    f32);
  RANDOM_DECL_SEQ(iris_double, normal, double,   f64);

  RANDOM_DECL_SEQ(iris_uint8,  log_normal, uint8_t,  u8);
  RANDOM_DECL_SEQ(iris_uint16, log_normal, uint16_t, u16);
  RANDOM_DECL_SEQ(iris_uint32, log_normal, uint32_t, u32);
  RANDOM_DECL_SEQ(iris_uint64, log_normal, uint64_t, u64);
  RANDOM_DECL_SEQ(iris_int8,   log_normal, int8_t,   i8);
  RANDOM_DECL_SEQ(iris_int16,  log_normal, int16_t,  i16);
  RANDOM_DECL_SEQ(iris_int32,  log_normal, int32_t,  i32);
  RANDOM_DECL_SEQ(iris_int64,  log_normal, int64_t,  i64);
  RANDOM_DECL_SEQ(iris_float,  log_normal, float,    f32);
  RANDOM_DECL_SEQ(iris_double, log_normal, double,   f64);

  RANDOM_DECL_SEQ(iris_uint8,  uniform_sobol, uint8_t,  u8);
  RANDOM_DECL_SEQ(iris_uint16, uniform_sobol, uint16_t, u16);
  RANDOM_DECL_SEQ(iris_uint32, uniform_sobol, uint32_t, u32);
  RANDOM_DECL_SEQ(iris_uint64, uniform_sobol, uint64_t, u64);
  RANDOM_DECL_SEQ(iris_int8,   uniform_sobol, int8_t,   i8);
  RANDOM_DECL_SEQ(iris_int16,  uniform_sobol, int16_t,  i16);
  RANDOM_DECL_SEQ(iris_int32,  uniform_sobol, int32_t,  i32);
  RANDOM_DECL_SEQ(iris_int64,  uniform_sobol, int64_t,  i64);
  RANDOM_DECL_SEQ(iris_float,  uniform_sobol, float,    f32);
  RANDOM_DECL_SEQ(iris_double, uniform_sobol, double,   f64);

  RANDOM_DECL_SEQ(iris_uint8,  normal_sobol, uint8_t,  u8);
  RANDOM_DECL_SEQ(iris_uint16, normal_sobol, uint16_t, u16);
  RANDOM_DECL_SEQ(iris_uint32, normal_sobol, uint32_t, u32);
  RANDOM_DECL_SEQ(iris_uint64, normal_sobol, uint64_t, u64);
  RANDOM_DECL_SEQ(iris_int8,   normal_sobol, int8_t,   i8);
  RANDOM_DECL_SEQ(iris_int16,  normal_sobol, int16_t,  i16);
  RANDOM_DECL_SEQ(iris_int32,  normal_sobol, int32_t,  i32);
  RANDOM_DECL_SEQ(iris_int64,  normal_sobol, int64_t,  i64);
  RANDOM_DECL_SEQ(iris_float,  normal_sobol, float,    f32);
  RANDOM_DECL_SEQ(iris_double, normal_sobol, double,   f64);

  RANDOM_DECL_SEQ(iris_uint8,  log_normal_sobol, uint8_t,  u8);
  RANDOM_DECL_SEQ(iris_uint16, log_normal_sobol, uint16_t, u16);
  RANDOM_DECL_SEQ(iris_uint32, log_normal_sobol, uint32_t, u32);
  RANDOM_DECL_SEQ(iris_uint64, log_normal_sobol, uint64_t, u64);
  RANDOM_DECL_SEQ(iris_int8,   log_normal_sobol, int8_t,   i8);
  RANDOM_DECL_SEQ(iris_int16,  log_normal_sobol, int16_t,  i16);
  RANDOM_DECL_SEQ(iris_int32,  log_normal_sobol, int32_t,  i32);
  RANDOM_DECL_SEQ(iris_int64,  log_normal_sobol, int64_t,  i64);
  RANDOM_DECL_SEQ(iris_float,  log_normal_sobol, float,    f32);
  RANDOM_DECL_SEQ(iris_double, log_normal_sobol, double,   f64);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H */


