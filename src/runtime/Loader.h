#ifndef IRIS_SRC_RT_LOADER_H
#define IRIS_SRC_RT_LOADER_H

#include <iris/iris.h>
#include <dlfcn.h>
typedef char (*c_string_array)[256] ;
typedef void (*__iris_kernel_ptr)();

#ifndef ENABLE_STATIC_LINKING
#define LOADFUNC(FUNC)          *(void**) (&FUNC) = dlsym(handle_, #FUNC);      \
                                if (!FUNC) _error("%s", dlerror());
#define LOADFUNC_OPTIONAL(FUNC) *(void**) (&FUNC) = dlsym(handle_, #FUNC);      
#define LOADFUNCSYM(FUNC, SYM)  *(void**) (&FUNC) = dlsym(handle_, #SYM);       \
                                if (!FUNC) _error("%s", dlerror());
#define LOADFUNCSYM_OPTIONAL(FUNC, SYM)  *(void**) (&FUNC) = dlsym(handle_, #SYM);       
#define LOADFUNCSILENT(FUNC)    *(void**) (&FUNC) = dlsym(handle_, #FUNC);
#define LOADFUNCEXT(FUNC)       *(void**) (&FUNC) = dlsym(handle_ext_, #FUNC);  \
                                if (!FUNC) _error("%s", dlerror());
#else
#define LOADFUNC(FUNC)                   FUNC = CLinkage::FUNC;
#define LOADFUNC_OPTIONAL(FUNC)          FUNC = CLinkage::FUNC;
#define LOADFUNCSYM(FUNC, SYM)           FUNC = CLinkage::SYM;
#define LOADFUNCSYM_OPTIONAL(FUNC, SYM)  FUNC = CLinkage::SYM;
#define LOADFUNCSILENT(FUNC)             FUNC = CLinkage::FUNC;
#define LOADFUNCEXT(FUNC)                FUNC = CLinkage::FUNC;
#endif
namespace iris {
namespace rt {

class Loader {
public:
  Loader();
  virtual ~Loader();

  int Load();
  virtual const char* library_precheck() { return NULL; }
  virtual const char* library() = 0;
  virtual int LoadFunctions();
  void (*iris_set_kernel_ptr_with_obj)(void *obj, __iris_kernel_ptr ptr);
  c_string_array (*iris_get_kernel_names)();
  bool IsFunctionExists(const char *name);
  virtual void *GetSymbol(const char *name);
  virtual void *GetFunctionSymbol(const char *name);
  void *GetFunctionPtr(const char *name) { return GetFunctionSymbol(name); }
  int SetKernelPtr(void *obj, const char *kernel_name);
  int LoadExtHandle(const char *libname);
  void *handle() { return handle_; }
  void enable_strict_handle_check() { strict_handle_check_ = true; }
private:
  int LoadHandle();

protected:
  void* handle_;
  void* handle_ext_;
  bool strict_handle_check_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_H */

