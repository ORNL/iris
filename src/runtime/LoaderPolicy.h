#ifndef IRIS_SRC_RT_LOADER_POLICY_H
#define IRIS_SRC_RT_LOADER_POLICY_H

#include "Loader.h"

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

private:
  char lib_[64];
  char name_[64]; 

  void* (*instance_)();
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_POLICY_H */

