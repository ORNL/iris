#include "LoaderQIREE.h"
#include "Debug.h"
#include "Platform.h"

namespace iris {
namespace rt {

LoaderQIREE::LoaderQIREE() : HostInterfaceClass("LIB_QIREE") {
}

LoaderQIREE::~LoaderQIREE() {
}
const char * LoaderQIREE::library() {
    char* path = NULL;
    Platform::GetPlatform()->EnvironmentGet("LIB_QIREE", &path, NULL);
    return path;
}
int LoaderQIREE::LoadFunctions() {
  LOADFUNC(qiree_create);
  LOADFUNC(qiree_destroy);
  LOADFUNC(qiree_load_module_from_memory);
  LOADFUNC(qiree_load_module_from_file);
  LOADFUNC(qiree_num_classical_reg);
  LOADFUNC(qiree_num_quantum_reg);
  LOADFUNC(qiree_sizeof_result);
  LOADFUNC(qiree_setup_executor);
  LOADFUNC(qiree_execute);
  LOADFUNC(qiree_save_result);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

