#ifndef IRIS_SRC_RT_LOADER_HOST2HIP_H
#define IRIS_SRC_RT_LOADER_HOST2HIP_H

#include "Loader.h"
#include "HostInterface.h"

namespace iris {
namespace rt {

class LoaderHost2HIP : public HostInterfaceClass {
public:
  LoaderHost2HIP();
  ~LoaderHost2HIP();

  int LoadFunctions();
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2HIP_H */

