#ifndef IRIS_SRC_RT_LOADER_HOST2OpenCL_H
#define IRIS_SRC_RT_LOADER_HOST2OpenCL_H

#include "Loader.h"

#include "HostInterface.h"

namespace iris {
namespace rt {

class LoaderHost2OpenCL : public HostInterfaceClass{
public:
  LoaderHost2OpenCL(const char *suffix);
  ~LoaderHost2OpenCL();

  const char* library();
  int LoadFunctions();
private:
  char *suffix_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2OpenCL_H */

