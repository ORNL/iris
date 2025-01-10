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
                template <typename T>
                static void Fill(T* array_ptr, size_t n_elements, T value);
                static void Fill(void *array_ptr, size_t n_elements, int element_type, ResetData &value);
                template <typename T>
                static void ArithSequence(T* array_ptr, size_t n_elements, T start, T increment);
                static void ArithSequence(void *array_ptr, size_t n_elements, int element_type, ResetData &value);
                template <typename T>
                static void GeometricSequence(T* array_ptr, size_t n_elements, T start, T ratio);
                static void GeometricSequence(void *array_ptr, size_t n_elements, int element_type, ResetData &value);
                template <typename T>
                static void RandomUniformSeq(T *arr, long long seed, size_t size, T p1, T p2);
                static void RandomUniformSeq(void *array_ptr, size_t size, int element_type, ResetData &reset);

                template <typename T>
                static void RandomNormalSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream);

                template <typename T>
                static void RandomLogNormalSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream);

                template <typename T>
                static void RandomUniformSobolSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream);

                template <typename T>
                static void RandomNormalSobolSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream);

                template <typename T>
                static void RandomLogNormalSobolSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream);

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
