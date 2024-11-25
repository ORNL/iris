#include "Debug.h"
#ifndef __APPLE__
#include <link.h>
#endif
#include <dlfcn.h>
#include <stdlib.h>

#include "Loader.h"

namespace CLinkage {
    extern "C" {
        void iris_set_kernel_ptr_with_obj(void *obj, __iris_kernel_ptr ptr);
        c_string_array iris_get_kernel_names();
    }
}
namespace iris {
namespace rt {

Loader::Loader() {
  handle_ = NULL;
  handle_ext_ = NULL;
  strict_handle_check_ = false;
  iris_get_kernel_names = NULL;
  iris_set_kernel_ptr_with_obj = NULL;
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

int Loader::LoadExtHandle(const char *libname) {
#ifndef DISABLE_DYNAMIC_LINKING
  handle_ext_ = dlopen(libname, RTLD_GLOBAL | RTLD_NOW);
  if (handle_ext_) {
#if 0
    struct link_map *map;
    dlinfo(handle_ext_, RTLD_DI_LINKMAP, &map);
    _trace("shared library path[%s]", realpath(map->l_name, NULL));
#endif
  } else {
    _warning("%s", dlerror());
    return IRIS_ERROR;
  }
#else
  _info("Dynamic linking is disabled. Skipped loading of library:%s\n", library());
#endif
  return IRIS_SUCCESS;
}

int Loader::LoadHandle() {
#ifndef DISABLE_DYNAMIC_LINKING
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
    _warning("%s", dlerror());
    return IRIS_ERROR;
  }
#else
  _info("Dynamic linking is disabled. Skipped loading of library:%s\n", library());
#endif
  return IRIS_SUCCESS;
}

int Loader::LoadFunctions() {
    LOADFUNC_OPTIONAL(iris_get_kernel_names);
    LOADFUNC_OPTIONAL(iris_set_kernel_ptr_with_obj);
    return IRIS_SUCCESS;
}

void *Loader::GetSymbol(const char *symbol_name) {
    if (strict_handle_check_ && handle_ == NULL) return NULL;
    void *kptr = dlsym(handle_, symbol_name);
    return kptr;
}

void *Loader::GetFunctionSymbol(const char *symbol_name) {
    if (strict_handle_check_ && handle_ == NULL) return NULL;
    void *kptr = dlsym(handle_, symbol_name);
    return kptr;
}

bool Loader::IsFunctionExists(const char *kernel_name) {
    if (strict_handle_check_ && handle_ == NULL) return false;
    __iris_kernel_ptr kptr;
    kptr = (__iris_kernel_ptr) dlsym(handle_, kernel_name);
    if (kptr == NULL) return false;
    return true;
}

int Loader::SetKernelPtr(void *obj, const char *kernel_name)
{
    if (iris_set_kernel_ptr_with_obj) {
        if (strict_handle_check_ && handle_ == NULL) return IRIS_ERROR;
        __iris_kernel_ptr kptr;
        kptr = (__iris_kernel_ptr) dlsym(handle_, kernel_name);
        iris_set_kernel_ptr_with_obj(obj, kptr);
        if (kptr != NULL) return IRIS_SUCCESS;
    }
    return IRIS_ERROR;
}

} /* namespace rt */
} /* namespace iris */

