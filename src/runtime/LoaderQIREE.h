#ifndef IRIS_SRC_RT_LOADER_QIREE_H
#define IRIS_SRC_RT_LOADER_QIREE_H

#include "Loader.h"
#include "HostInterface.h"
//#include <iris/qiree/qiree.h>

namespace iris {
namespace rt {

class LoaderQIREE : public HostInterfaceClass {
public:
  LoaderQIREE();
  ~LoaderQIREE();

  //const char* library_precheck() { return "cuInit"; }
  const char* library();// { return "libqiree.so"; }
  int LoadFunctions();
  void (*parse_input_c)(int, char **);

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_QIREE_H */

