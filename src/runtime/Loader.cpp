#include "Loader.h"
#include "Debug.h"
#include <link.h>
#include <dlfcn.h>
#include <stdlib.h>

namespace brisbane {
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
  if (!library()) return BRISBANE_OK;
  if (LoadHandle() != BRISBANE_OK) return BRISBANE_ERR;
  return LoadFunctions();
}

int Loader::LoadHandle() {
  if (library_precheck() && dlsym(RTLD_DEFAULT, library_precheck())) {
    handle_ = RTLD_DEFAULT;
    return BRISBANE_OK;
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
    return BRISBANE_ERR;
  }
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

