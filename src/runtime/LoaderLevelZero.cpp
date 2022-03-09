#include "LoaderLevelZero.h"
#include "Debug.h"

namespace iris {
namespace rt {

LoaderLevelZero::LoaderLevelZero() {
  pthread_mutex_init(&mutex_, NULL);
}

LoaderLevelZero::~LoaderLevelZero() {
  pthread_mutex_destroy(&mutex_);
}

int LoaderLevelZero::LoadFunctions() {
  LOADFUNC(zeInit);
  LOADFUNC(zeDriverGet);
  LOADFUNC(zeDeviceGet);
  LOADFUNC(zeDeviceGetProperties);
  LOADFUNC(zeContextCreate);
  LOADFUNC(zeContextDestroy);
  LOADFUNC(zeMemAllocDevice);
  LOADFUNC(zeMemFree);
  LOADFUNC(zeCommandQueueCreate);
  LOADFUNC(zeCommandListCreate);
  LOADFUNC(zeCommandListCreateImmediate);
  LOADFUNC(zeCommandListReset);
  LOADFUNC(zeCommandListDestroy);
  LOADFUNC(zeCommandListAppendMemoryCopy);
  LOADFUNC(zeCommandListAppendLaunchKernel);
  LOADFUNC(zeCommandListAppendSignalEvent);
  LOADFUNC(zeCommandQueueExecuteCommandLists);
  LOADFUNC(zeCommandQueueSynchronize);
  LOADFUNC(zeModuleCreate);
  LOADFUNC(zeKernelCreate);
  LOADFUNC(zeKernelSetArgumentValue);
  LOADFUNC(zeFenceCreate);
  LOADFUNC(zeFenceDestroy);
  LOADFUNC(zeEventPoolCreate);
  LOADFUNC(zeEventCreate);
  LOADFUNC(zeEventDestroy);
  LOADFUNC(zeEventHostSynchronize);
  return IRIS_SUCCESS;
}

void LoaderLevelZero::Lock() {
  pthread_mutex_lock(&mutex_);
}

void LoaderLevelZero::Unlock() {
  pthread_mutex_unlock(&mutex_);
}

} /* namespace rt */
} /* namespace iris */

