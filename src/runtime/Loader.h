#ifndef BRISBANE_SRC_RT_LOADER_H
#define BRISBANE_SRC_RT_LOADER_H

#include <brisbane/brisbane.h>
#include <dlfcn.h>

#define LOADFUNC(FUNC)          *(void**) (&FUNC) = dlsym(handle_, #FUNC);      \
                                if (!FUNC) _error("%s", dlerror())
#define LOADFUNCSYM(FUNC, SYM)  *(void**) (&FUNC) = dlsym(handle_, #SYM);       \
                                if (!FUNC) _error("%s", dlerror())
#define LOADFUNCSILENT(FUNC)    *(void**) (&FUNC) = dlsym(handle_, #FUNC);
#define LOADFUNCEXT(FUNC)       *(void**) (&FUNC) = dlsym(handle_ext_, #FUNC);  \
                                if (!FUNC) _error("%s", dlerror())

namespace brisbane {
namespace rt {

class Loader {
public:
  Loader();
  virtual ~Loader();

  int Load();
  virtual const char* library_precheck() { return NULL; }
  virtual const char* library() = 0;
  virtual int LoadFunctions() = 0;

private:
  int LoadHandle();

protected:
  void* handle_;
  void* handle_ext_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_LOADER_H */

