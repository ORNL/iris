#include "LoaderPolicy.h"
#include "Policy.h"
#include "Debug.h"

namespace iris {
namespace rt {

LoaderPolicy::LoaderPolicy(const char* lib, const char* name) : Loader() {
  name_ = std::string(name);
  lib_ = std::string(lib);
}

LoaderPolicy::~LoaderPolicy() {
}

Policy* LoaderPolicy::policy() {
  return (Policy*) (instance_)();
}

const char* LoaderPolicy::library() {
  return lib();
}

int LoaderPolicy::LoadFunctions() {
  char func[128];
  sprintf(func, "%s_instance", name());
  *(void**) (&instance_) = dlsym(handle_, func);
  if (!instance_) {
    _error("%s", dlerror());
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

void LoaderPolicy::Init(void* arg) {
  ((Policy*) (instance_)())->Init(arg);
}

} /* namespace rt */
} /* namespace iris */

