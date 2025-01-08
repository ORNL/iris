#include "LoaderDefaultKernel.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {


LoaderDefaultKernel::LoaderDefaultKernel(const char *path) : HostInterfaceClass(path) {
    enable_strict_handle_check();
}

LoaderDefaultKernel::~LoaderDefaultKernel() {
}

int LoaderDefaultKernel::LoadFunctions() {
  HostInterfaceClass::LoadFunctions();
  REGISTER_HOST_WRAPPER(iris_reset_i64, iris_reset_i64);
  REGISTER_HOST_WRAPPER(iris_reset_i32, iris_reset_i32);
  REGISTER_HOST_WRAPPER(iris_reset_i16, iris_reset_i16);
  REGISTER_HOST_WRAPPER(iris_reset_i8, iris_reset_i8);
  REGISTER_HOST_WRAPPER(iris_reset_u64, iris_reset_u64);
  REGISTER_HOST_WRAPPER(iris_reset_u32, iris_reset_u32);
  REGISTER_HOST_WRAPPER(iris_reset_u16, iris_reset_u16);
  REGISTER_HOST_WRAPPER(iris_reset_u8, iris_reset_u8);
  REGISTER_HOST_WRAPPER(iris_reset_float, iris_reset_float);
  REGISTER_HOST_WRAPPER(iris_reset_double, iris_reset_double);

  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_i64, iris_arithmetic_seq_i64);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_i32, iris_arithmetic_seq_i32);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_i16, iris_arithmetic_seq_i16);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_i8, iris_arithmetic_seq_i8);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_u64, iris_arithmetic_seq_u64);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_u32, iris_arithmetic_seq_u32);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_u16, iris_arithmetic_seq_u16);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_u8, iris_arithmetic_seq_u8);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_float, iris_arithmetic_seq_float);
  REGISTER_HOST_WRAPPER(iris_arithmetic_seq_double, iris_arithmetic_seq_double);

  REGISTER_HOST_WRAPPER(iris_geometric_seq_i64, iris_geometric_seq_i64);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_i32, iris_geometric_seq_i32);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_i16, iris_geometric_seq_i16);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_i8, iris_geometric_seq_i8);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_u64, iris_geometric_seq_u64);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_u32, iris_geometric_seq_u32);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_u16, iris_geometric_seq_u16);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_u8, iris_geometric_seq_u8);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_float, iris_geometric_seq_float);
  REGISTER_HOST_WRAPPER(iris_geometric_seq_double, iris_geometric_seq_double);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


