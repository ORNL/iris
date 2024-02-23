#ifndef IRIS_SRC_RT_LOADER_HOST2CUDA_H
#define IRIS_SRC_RT_LOADER_HOST2CUDA_H

#include "Loader.h"

#include "HostInterface.h"

namespace iris {
namespace rt {

class LoaderHost2CUDA : public HostInterfaceClass {
public:
  LoaderHost2CUDA();
  ~LoaderHost2CUDA();

  int LoadFunctions();
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_HOST2CUDA_H */

