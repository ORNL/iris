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
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

#include <cstdint>
#include <utility>

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
                static const char *GetTypeString(int type) {
                    switch(type) {
                        case iris_float: return "float";
                        case iris_double: return "double";
                        case iris_uint64: return "uint64";
                        case iris_uint32: return "uint32";
                        case iris_uint16: return "uint16";
                        case iris_uint8: return "uint8";
                        case iris_int64: return "int64";
                        case iris_int32: return "int32";
                        case iris_int16: return "int16";
                        case iris_int8: return "int8";
                        case iris_int: return "int";
                        case iris_uint: return "uint";
                        default: return "error";
                    }
                }
                template<class T>
                static void insert_at(std::vector<T>& v, std::size_t pos, const T& value) {
                    if (pos > v.size()) throw std::out_of_range("pos out of range");
                    v.insert(v.begin() + pos, value);   // shifts [pos..end) to the right by 1
                }
                static bool IsSizesSame( const size_t *src_size,   
                        const size_t *dst_size, int dims);              
                static bool IsTileContiguous( const size_t *strides,   
                        const size_t *tile_size, int dims);              

#if defined(_WIN32)
                static size_t PageSize() { 
                    SYSTEM_INFO si; GetSystemInfo(&si); return (size_t)si.dwPageSize; 
                }
#else
                static size_t PageSize() { 
                    return (size_t)sysconf(_SC_PAGESIZE); 
                }
#endif
                static inline std::pair<void*, size_t>
                    PageAligned(void* host, size_t bytes)
                    {
                        const size_t pgsz  = PageSize(); // your function: e.g., sysconf(_SC_PAGESIZE)
                        const uintptr_t p  = reinterpret_cast<uintptr_t>(host);
                        const uintptr_t b  = p & ~(static_cast<uintptr_t>(pgsz) - 1u); // align down
                        const size_t    d  = static_cast<size_t>(p - b);               // head fragment
                        const size_t    n  = bytes + d;                                // bytes to cover
                        const size_t    regsz = (n + (pgsz - 1)) & ~(pgsz - 1);        // round up to page

                        return { reinterpret_cast<void*>(b), regsz };
                    }
                static inline void*
                    PageAligned(void* host)
                    {
                        const size_t pgsz  = PageSize(); // your function: e.g., sysconf(_SC_PAGESIZE)
                        const uintptr_t p  = reinterpret_cast<uintptr_t>(host);
                        const uintptr_t b  = p & ~(static_cast<uintptr_t>(pgsz) - 1u); // align down
                        return (void *)b;
                    }

        };

    } /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_UTILS_H */
