#include "Loader.h"
#include "Debug.h"
#ifndef __APPLE__
#include <link.h>
#endif
#include <dlfcn.h>
#include <stdlib.h>

namespace iris {
namespace rt {

Loader::Loader() {
  handle_ = NULL;
  handle_ext_ = NULL;
}

Loader::~Loader() {
  if (handle_) if (dlclose(handle_) != 0) _error("%s", dlerror());
  if (handle_ext_) if (dlclose(handle_ext_) != 0) _error("%s", dlerror());
}

int Loader::Load() {
  if (!library()) return IRIS_OK;
  if (LoadHandle() != IRIS_OK) return IRIS_ERR;
  return LoadFunctions();
}

int Loader::LoadHandle() {
  if (library_precheck() && dlsym(RTLD_DEFAULT, library_precheck())) {
    handle_ = RTLD_DEFAULT;
    return IRIS_OK;
  }
  handle_ = dlopen(library(), RTLD_GLOBAL | RTLD_NOW);
  if (handle_) {
#if 0
    struct link_map *map;
    dlinfo(handle_, RTLD_DI_LINKMAP, &map);
    _trace("shared library path[%s]", realpath(map->l_name, NULL));
#endif
  } else {
    _trace("%s", dlerror());
    return IRIS_ERR;
  }
  return IRIS_OK;
}

} /* namespace rt */
} /* namespace iris */

