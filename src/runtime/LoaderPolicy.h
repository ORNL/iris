#ifndef IRIS_SRC_RT_LOADER_POLICY_H
#define IRIS_SRC_RT_LOADER_POLICY_H

#include "Loader.h"
#include <string>

namespace iris {
namespace rt {

class Policy;

class LoaderPolicy : public Loader {
public:
  LoaderPolicy(const char* lib, const char* name);
  ~LoaderPolicy();

  Policy* policy();

  const char* library();
  int LoadFunctions();
  void Init(void* arg);
  const char *name() { return name_.c_str(); }
  const char *lib() { return lib_.c_str(); }

private:
  std::string lib_;
  std::string name_;

  void* (*instance_)();
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_POLICY_H */

