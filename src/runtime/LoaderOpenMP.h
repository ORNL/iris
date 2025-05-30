#ifndef IRIS_SRC_RT_LOADER_OPENMP_H
#define IRIS_SRC_RT_LOADER_OPENMP_H

#include "Loader.h"
#include "HostInterface.h"

namespace iris {
namespace rt {

class LoaderOpenMP : public HostInterfaceClass {
public:
  LoaderOpenMP();
  ~LoaderOpenMP();

  int LoadFunctions();
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_OPENMP_H */

