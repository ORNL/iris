#ifndef IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H
#define IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H

#include "Loader.h"

#include "HostInterface.h"
#include <string>

using namespace std;

namespace iris {
namespace rt {

class LoaderDefaultKernel : public HostInterfaceLoader {
public:
  LoaderDefaultKernel(string path);
  ~LoaderDefaultKernel();

  const char* library();
  int LoadFunctions();
#define RESET_DECL_SEQ(T, M)  void (*iris_reset_ ## M)(T *arr, T value, size_t size, void *stream) ;
  RESET_DECL_SEQ(uint8_t,  u8);
  RESET_DECL_SEQ(uint16_t, u16);
  RESET_DECL_SEQ(uint32_t, u32);
  RESET_DECL_SEQ(uint64_t, u64);
  RESET_DECL_SEQ(int8_t,   i8);
  RESET_DECL_SEQ(int16_t,  i16);
  RESET_DECL_SEQ(int32_t,  i32);
  RESET_DECL_SEQ(int64_t,  i64);
  RESET_DECL_SEQ(float,    f32);
  RESET_DECL_SEQ(double,   f64);

#define ARITH_DECL_SEQ(T, M)  void (*iris_arithmetic_seq_ ## M)(T *arr, T start, T step, size_t size, void *stream) ;
  ARITH_DECL_SEQ(uint8_t,  u8);
  ARITH_DECL_SEQ(uint16_t, u16);
  ARITH_DECL_SEQ(uint32_t, u32);
  ARITH_DECL_SEQ(uint64_t, u64);
  ARITH_DECL_SEQ(int8_t,   i8);
  ARITH_DECL_SEQ(int16_t,  i16);
  ARITH_DECL_SEQ(int32_t,  i32);
  ARITH_DECL_SEQ(int64_t,  i64);
  ARITH_DECL_SEQ(float,    f32);
  ARITH_DECL_SEQ(double,   f64);

#define GEOM_DECL_SEQ(T, M)  void (*iris_geometric_seq_ ## M)(T *arr, T start, T step, size_t size, void *stream) ;
  GEOM_DECL_SEQ(uint8_t,  u8);
  GEOM_DECL_SEQ(uint16_t, u16);
  GEOM_DECL_SEQ(uint32_t, u32);
  GEOM_DECL_SEQ(uint64_t, u64);
  GEOM_DECL_SEQ(int8_t,   i8);
  GEOM_DECL_SEQ(int16_t,  i16);
  GEOM_DECL_SEQ(int32_t,  i32);
  GEOM_DECL_SEQ(int64_t,  i64);
  GEOM_DECL_SEQ(float,    f32);
  GEOM_DECL_SEQ(double,   f64);
#define RANDOM_DECL_SEQ(RTYPE, T, TAG)  void (*iris_random_ ## RTYPE ## _seq_ ## TAG)(T *arr, unsigned long long seed, T p1, T p2, size_t size, void *stream);
  RANDOM_DECL_SEQ(uniform, uint8_t,  u8);
  RANDOM_DECL_SEQ(uniform, uint16_t, u16);
  RANDOM_DECL_SEQ(uniform, uint32_t, u32);
  RANDOM_DECL_SEQ(uniform, uint64_t, u64);
  RANDOM_DECL_SEQ(uniform, int8_t,   i8);
  RANDOM_DECL_SEQ(uniform, int16_t,  i16);
  RANDOM_DECL_SEQ(uniform, int32_t,  i32);
  RANDOM_DECL_SEQ(uniform, int64_t,  i64);
  RANDOM_DECL_SEQ(uniform, float,    f32);
  RANDOM_DECL_SEQ(uniform, double,   f64);

  RANDOM_DECL_SEQ(normal, uint8_t,  u8);
  RANDOM_DECL_SEQ(normal, uint16_t, u16);
  RANDOM_DECL_SEQ(normal, uint32_t, u32);
  RANDOM_DECL_SEQ(normal, uint64_t, u64);
  RANDOM_DECL_SEQ(normal, int8_t,   i8);
  RANDOM_DECL_SEQ(normal, int16_t,  i16);
  RANDOM_DECL_SEQ(normal, int32_t,  i32);
  RANDOM_DECL_SEQ(normal, int64_t,  i64);
  RANDOM_DECL_SEQ(normal, float,    f32);
  RANDOM_DECL_SEQ(normal, double,   f64);

  RANDOM_DECL_SEQ(log_normal, uint8_t,  u8);
  RANDOM_DECL_SEQ(log_normal, uint16_t, u16);
  RANDOM_DECL_SEQ(log_normal, uint32_t, u32);
  RANDOM_DECL_SEQ(log_normal, uint64_t, u64);
  RANDOM_DECL_SEQ(log_normal, int8_t,   i8);
  RANDOM_DECL_SEQ(log_normal, int16_t,  i16);
  RANDOM_DECL_SEQ(log_normal, int32_t,  i32);
  RANDOM_DECL_SEQ(log_normal, int64_t,  i64);
  RANDOM_DECL_SEQ(log_normal, float,    f32);
  RANDOM_DECL_SEQ(log_normal, double,   f64);

  RANDOM_DECL_SEQ(uniform_sobol, uint8_t,  u8);
  RANDOM_DECL_SEQ(uniform_sobol, uint16_t, u16);
  RANDOM_DECL_SEQ(uniform_sobol, uint32_t, u32);
  RANDOM_DECL_SEQ(uniform_sobol, uint64_t, u64);
  RANDOM_DECL_SEQ(uniform_sobol, int8_t,   i8);
  RANDOM_DECL_SEQ(uniform_sobol, int16_t,  i16);
  RANDOM_DECL_SEQ(uniform_sobol, int32_t,  i32);
  RANDOM_DECL_SEQ(uniform_sobol, int64_t,  i64);
  RANDOM_DECL_SEQ(uniform_sobol, float,    f32);
  RANDOM_DECL_SEQ(uniform_sobol, double,   f64);

  RANDOM_DECL_SEQ(normal_sobol, uint8_t,  u8);
  RANDOM_DECL_SEQ(normal_sobol, uint16_t, u16);
  RANDOM_DECL_SEQ(normal_sobol, uint32_t, u32);
  RANDOM_DECL_SEQ(normal_sobol, uint64_t, u64);
  RANDOM_DECL_SEQ(normal_sobol, int8_t,   i8);
  RANDOM_DECL_SEQ(normal_sobol, int16_t,  i16);
  RANDOM_DECL_SEQ(normal_sobol, int32_t,  i32);
  RANDOM_DECL_SEQ(normal_sobol, int64_t,  i64);
  RANDOM_DECL_SEQ(normal_sobol, float,    f32);
  RANDOM_DECL_SEQ(normal_sobol, double,   f64);

  RANDOM_DECL_SEQ(log_normal_sobol, uint8_t,  u8);
  RANDOM_DECL_SEQ(log_normal_sobol, uint16_t, u16);
  RANDOM_DECL_SEQ(log_normal_sobol, uint32_t, u32);
  RANDOM_DECL_SEQ(log_normal_sobol, uint64_t, u64);
  RANDOM_DECL_SEQ(log_normal_sobol, int8_t,   i8);
  RANDOM_DECL_SEQ(log_normal_sobol, int16_t,  i16);
  RANDOM_DECL_SEQ(log_normal_sobol, int32_t,  i32);
  RANDOM_DECL_SEQ(log_normal_sobol, int64_t,  i64);
  RANDOM_DECL_SEQ(log_normal_sobol, float,    f32);
  RANDOM_DECL_SEQ(log_normal_sobol, double,   f64);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H */


