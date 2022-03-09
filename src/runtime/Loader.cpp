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
  if (!library()) return IRIS_SUCCESS;
  if (LoadHandle() != IRIS_SUCCESS) return IRIS_ERROR;
  return LoadFunctions();
}

int Loader::LoadHandle() {
  if (library_precheck() && dlsym(RTLD_DEFAULT, library_precheck())) {
    handle_ = RTLD_DEFAULT;
    return IRIS_SUCCESS;
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
    return IRIS_ERROR;
  }
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

