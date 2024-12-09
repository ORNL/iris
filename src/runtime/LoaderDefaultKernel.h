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
  void (*iris_reset_i64)(int64_t *arr, int64_t value, size_t size, void *stream) ;
  void (*iris_reset_i32)(int32_t *arr, int32_t value, size_t size, void *stream) ;
  void (*iris_reset_i16)(int16_t *arr, int16_t value, size_t size, void *stream) ;
  void (*iris_reset_i8)(int8_t  *arr, int8_t value,  size_t size, void *stream) ;
  void (*iris_reset_u64)(uint64_t *arr, uint64_t value, size_t size, void *stream) ;
  void (*iris_reset_u32)(uint32_t *arr, uint32_t value, size_t size, void *stream) ;
  void (*iris_reset_u16)(uint16_t *arr, uint16_t value, size_t size, void *stream) ;
  void (*iris_reset_u8)(uint8_t  *arr, uint8_t value,  size_t size, void *stream) ;
  void (*iris_reset_float)(float *arr, float value, size_t size, void *stream) ;
  void (*iris_reset_double)(double *arr, double value, size_t size, void *stream) ;


  void (*iris_arithmetic_seq_i64)(int64_t *arr, int64_t start, int64_t increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_i32)(int32_t *arr, int32_t start, int32_t increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_i16)(int16_t *arr, int16_t start, int16_t increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_i8)(int8_t  *arr, int8_t start, int8_t increment,  size_t size, void *stream) ;
  void (*iris_arithmetic_seq_u64)(uint64_t *arr, uint64_t start, uint64_t increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_u32)(uint32_t *arr, uint32_t start, uint32_t increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_u16)(uint16_t *arr, uint16_t start, uint16_t increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_u8)(uint8_t  *arr, uint8_t start, uint8_t increment,  size_t size, void *stream) ;
  void (*iris_arithmetic_seq_float)(float *arr, float start, float increment, size_t size, void *stream) ;
  void (*iris_arithmetic_seq_double)(double *arr, double start, double increment, size_t size, void *stream) ;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_DEFAULT_KERNEL_H */


