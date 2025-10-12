#ifndef IRIS_SRC_RT_UTILS_H
#define IRIS_SRC_RT_UTILS_H

#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <set>
#include <map>
using namespace std;
namespace iris {
    namespace rt {

        class Utils {
            public:
                static void Logo(bool color);
                static string GetBackendName(int backend_id);
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

                static bool Exist(const char* path);
                static long Mtime(char* path);
                static void Datetime(char* str);
                static void PrintStackTrace();
                static int  CPUCoresCount();
                static void SetThreadAffinity(unsigned int core_id);
                static void MemCpy3D(uint8_t *dev, uint8_t *host, size_t *off, size_t *dev_sizes, size_t *host_sizes, size_t elem_size, bool host_2_dev=true);
                static void MemCpy(int dim, 
                        void* dst, const size_t *dst_off, const size_t* dst_strides, 
                        const void* src, const size_t* src_off, const size_t* src_strides, 
                        const size_t* tile_size, size_t elem_size);
                static void ReadSet(std::set<std::string> & out, const char *data);
                static void ReadVector(std::vector<std::string> & out, const char *data, const char *delim=NULL);
                static void ReadMap(std::map<std::string, int> & in_map, std::map<int, bool> & out_map, const char *data);
                static string GetPtrString(void **data, int dim) 
                {
                    stringstream ss;
                    ss << "[";
                    for (int i = 0; i < dim; i++) {
                        ss << hex << data[i];
                        if (i != dim-1) 
                            ss << ", ";
                    }
                    ss << "]";
                    return ss.str();
                }
                template <typename T>
                static string GetString(T *data, int dim) 
                {
                    stringstream ss;
                    ss << "[";
                    for (int i = 0; i < dim; i++) {
                        ss << data[i];
                        if (i != dim-1) 
                            ss << ", ";
                    }
                    ss << "]";
                    return ss.str();
                }
                template<class T>
                static void insert_at(std::vector<T>& v, std::size_t pos, const T& value) {
                    if (pos > v.size()) throw std::out_of_range("pos out of range");
                    v.insert(v.begin() + pos, value);   // shifts [pos..end) to the right by 1
                }
                static bool IsTileContiguous( const size_t strides[3],   
                        const size_t tile_size[3], int dims);              
        };

    } /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_UTILS_H */
