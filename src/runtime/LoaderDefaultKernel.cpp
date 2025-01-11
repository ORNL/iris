#include "LoaderDefaultKernel.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>

namespace iris {
namespace rt {


LoaderDefaultKernel::LoaderDefaultKernel(string path) : HostInterfaceLoader(path) {
    enable_strict_handle_check();
}

LoaderDefaultKernel::~LoaderDefaultKernel() {
}

const char* LoaderDefaultKernel::library() {
    return kernel_env().c_str();
}

int LoaderDefaultKernel::LoadFunctions() {
  HostInterfaceLoader::LoadFunctions();
  LOADFUNC(iris_reset_i64);
  LOADFUNC(iris_reset_i32);
  LOADFUNC(iris_reset_i16);
  LOADFUNC(iris_reset_i8);
  LOADFUNC(iris_reset_u64);
  LOADFUNC(iris_reset_u32);
  LOADFUNC(iris_reset_u16);
  LOADFUNC(iris_reset_u8);
  LOADFUNC(iris_reset_f32);
  LOADFUNC(iris_reset_f64);

  LOADFUNC(iris_arithmetic_seq_i64);
  LOADFUNC(iris_arithmetic_seq_i32);
  LOADFUNC(iris_arithmetic_seq_i16);
  LOADFUNC(iris_arithmetic_seq_i8);
  LOADFUNC(iris_arithmetic_seq_u64);
  LOADFUNC(iris_arithmetic_seq_u32);
  LOADFUNC(iris_arithmetic_seq_u16);
  LOADFUNC(iris_arithmetic_seq_u8);
  LOADFUNC(iris_arithmetic_seq_f32);
  LOADFUNC(iris_arithmetic_seq_f64);

  LOADFUNC(iris_geometric_seq_i64);
  LOADFUNC(iris_geometric_seq_i32);
  LOADFUNC(iris_geometric_seq_i16);
  LOADFUNC(iris_geometric_seq_i8);
  LOADFUNC(iris_geometric_seq_u64);
  LOADFUNC(iris_geometric_seq_u32);
  LOADFUNC(iris_geometric_seq_u16);
  LOADFUNC(iris_geometric_seq_u8);
  LOADFUNC(iris_geometric_seq_f32);
  LOADFUNC(iris_geometric_seq_f64);

  LOADFUNC(iris_random_uniform_seq_f32);
  LOADFUNC(iris_random_uniform_seq_f64);
  LOADFUNC(iris_random_uniform_seq_u64);
  LOADFUNC(iris_random_uniform_seq_u32);
  LOADFUNC(iris_random_uniform_seq_u16);
  LOADFUNC(iris_random_uniform_seq_u8);
  LOADFUNC(iris_random_uniform_seq_i64);
  LOADFUNC(iris_random_uniform_seq_i32);
  LOADFUNC(iris_random_uniform_seq_i16);
  LOADFUNC(iris_random_uniform_seq_i8);

  LOADFUNC(iris_random_normal_seq_f32);
  LOADFUNC(iris_random_normal_seq_f64);
  LOADFUNC(iris_random_normal_seq_u64);
  LOADFUNC(iris_random_normal_seq_u32);
  LOADFUNC(iris_random_normal_seq_u16);
  LOADFUNC(iris_random_normal_seq_u8);
  LOADFUNC(iris_random_normal_seq_i64);
  LOADFUNC(iris_random_normal_seq_i32);
  LOADFUNC(iris_random_normal_seq_i16);
  LOADFUNC(iris_random_normal_seq_i8);

  LOADFUNC(iris_random_log_normal_seq_f32);
  LOADFUNC(iris_random_log_normal_seq_f64);
  LOADFUNC(iris_random_log_normal_seq_u64);
  LOADFUNC(iris_random_log_normal_seq_u32);
  LOADFUNC(iris_random_log_normal_seq_u16);
  LOADFUNC(iris_random_log_normal_seq_u8);
  LOADFUNC(iris_random_log_normal_seq_i64);
  LOADFUNC(iris_random_log_normal_seq_i32);
  LOADFUNC(iris_random_log_normal_seq_i16);
  LOADFUNC(iris_random_log_normal_seq_i8);

  LOADFUNC(iris_random_uniform_sobol_seq_f32);
  LOADFUNC(iris_random_uniform_sobol_seq_f64);
  LOADFUNC(iris_random_uniform_sobol_seq_u64);
  LOADFUNC(iris_random_uniform_sobol_seq_u32);
  LOADFUNC(iris_random_uniform_sobol_seq_u16);
  LOADFUNC(iris_random_uniform_sobol_seq_u8);
  LOADFUNC(iris_random_uniform_sobol_seq_i64);
  LOADFUNC(iris_random_uniform_sobol_seq_i32);
  LOADFUNC(iris_random_uniform_sobol_seq_i16);
  LOADFUNC(iris_random_uniform_sobol_seq_i8);

  LOADFUNC(iris_random_normal_sobol_seq_f32);
  LOADFUNC(iris_random_normal_sobol_seq_f64);
  LOADFUNC(iris_random_normal_sobol_seq_u64);
  LOADFUNC(iris_random_normal_sobol_seq_u32);
  LOADFUNC(iris_random_normal_sobol_seq_u16);
  LOADFUNC(iris_random_normal_sobol_seq_u8);
  LOADFUNC(iris_random_normal_sobol_seq_i64);
  LOADFUNC(iris_random_normal_sobol_seq_i32);
  LOADFUNC(iris_random_normal_sobol_seq_i16);
  LOADFUNC(iris_random_normal_sobol_seq_i8);

  LOADFUNC(iris_random_log_normal_sobol_seq_f32);
  LOADFUNC(iris_random_log_normal_sobol_seq_f64);
  LOADFUNC(iris_random_log_normal_sobol_seq_u64);
  LOADFUNC(iris_random_log_normal_sobol_seq_u32);
  LOADFUNC(iris_random_log_normal_sobol_seq_u16);
  LOADFUNC(iris_random_log_normal_sobol_seq_u8);
  LOADFUNC(iris_random_log_normal_sobol_seq_i64);
  LOADFUNC(iris_random_log_normal_sobol_seq_i32);
  LOADFUNC(iris_random_log_normal_sobol_seq_i16);
  LOADFUNC(iris_random_log_normal_sobol_seq_i8);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */


