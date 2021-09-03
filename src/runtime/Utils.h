#ifndef BRISBANE_SRC_RT_UTILS_H
#define BRISBANE_SRC_RT_UTILS_H

#include <stdlib.h>

namespace brisbane {
namespace rt {

class Utils {
public:
static void Logo(bool color);
static int ReadFile(char* path, char** string, size_t* len);
static void Datetime(char* str);
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_UTILS_H */
