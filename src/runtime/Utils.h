#ifndef IRIS_SRC_RT_UTILS_H
#define IRIS_SRC_RT_UTILS_H

#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
using namespace std;
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
                static void PrintStackTrace();
                static int  CPUCoresCount();
                static void SetThreadAffinity(unsigned int core_id);
                static void MemCpy3D(uint8_t *dev, uint8_t *host, size_t *off, size_t *dev_sizes, size_t *host_sizes, size_t elem_size, bool host_2_dev=true);
                static void ReadSet(std::set<std::string> & out, const char *data);
                static void ReadVector(std::vector<std::string> & out, const char *data);
                static void ReadMap(std::map<std::string, int> & in_map, std::map<int, bool> & out_map, const char *data);
        };

    } /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_UTILS_H */
