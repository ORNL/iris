#include "LoaderQIREE.h"
#include "Debug.h"
#include "Platform.h"

namespace iris {
namespace rt {

LoaderQIREE::LoaderQIREE() : HostInterfaceClass("LIB_QIREE") {
}

LoaderQIREE::~LoaderQIREE() {
}
const char * LoaderQIREE::library() {
    char* path = NULL;
    Platform::GetPlatform()->EnvironmentGet("LIB_QIREE", &path, NULL);
    return path;
}
int LoaderQIREE::LoadFunctions() {
  LOADFUNC(parse_input_c);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

