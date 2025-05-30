#ifndef AIWC_H
#define AIWC_H

#include <stdlib.h>

namespace iris {
namespace rt {
namespace plugin {

class AIWC_Utils {
public:
static bool IsAIWCDevice(char* name,char* vendor);
static void SetEnvironment(const char *name, const char *value);
static bool HaveMetrics(char* path);
static const char* MetricLocation(char* digest);
static int ReadFile(char* path, char** string, size_t* len);
static char* ComputeFileDigest(char* path);
static char* ComputeDigest(char* src);
static bool MetricsForKernelFileExist(char* path);
};

} /* namespace plugin */
} /* namespace rt */
} /* namespace iris */

#endif /* AIWC_UTILS_H */
