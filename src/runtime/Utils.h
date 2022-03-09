#ifndef IRIS_SRC_RT_UTILS_H
#define IRIS_SRC_RT_UTILS_H

#include <stdlib.h>

namespace iris {
namespace rt {

class Utils {
public:
static void Logo(bool color);
static int ReadFile(char* path, char** string, size_t* len);
static int Mkdir(char* path);
static bool Exist(char* path);
static long Mtime(char* path);
static void Datetime(char* str);
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_UTILS_H */
