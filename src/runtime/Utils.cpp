#include <iris/iris.h>
#include "math.h"
#include "Utils.h"
#include "Config.h"
#include "Debug.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef ANDROID
#include <execinfo.h>
#endif
#include <cstdlib>
#include <regex>
#include <random>
#include <cxxabi.h>
#include <thread>
#include <sched.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif
namespace iris {
namespace rt {

// # Spec
// # Print backtrace of a function
void Utils::PrintStackTrace() {
#ifdef ANDROID
    printf("[Error] PrintStackTrace is not supported\n");
#else
    const int max_frames = 16;
    void* frame_addrs[max_frames];
    int num_frames = backtrace(frame_addrs, max_frames);

    char** symbols = backtrace_symbols(frame_addrs, num_frames);

    std::regex pattern1(".*\\(([a-zA-Z_0-9][a-zA-Z0-9_]*)\\).*");
    std::regex pattern2(".*\\(([a-zA-Z_0-9][a-zA-Z0-9_]*)[+](0x[a-zA-Z0-9]*)\\).*");
    std::smatch match;
    char func_name[1024];
    for (int i = 0; i < num_frames; ++i) {
        // Demangle C++ function names
        size_t func_name_size = 1024;
        int func_name_offset;
        int status;
        std::string symbol = symbols[i];
        std::string hex_str = "";
        unsigned long long hex_value = 0;
        if (std::regex_match(symbol, match, pattern1)) {
            symbol = match[1].str();
        }
        else if (std::regex_match(symbol, match, pattern2)) {
            symbol = match[1].str();
            hex_str = match[2].str();
            //char *end_ptr;
            //hex_value = strtoull(match[2].str().c_str(), &end_ptr, 16);
        }
        char* demangled = abi::__cxa_demangle(symbol.c_str(), func_name, &func_name_size, &status);
        if (status == 0) {
            std::cout << "[" << i << "] " << demangled << "("<<hex_str<<")"<< std::endl;
        } else {
            std::cout << "[" << i << "] " << symbols[i] << std::endl;
        }
    }

    free(symbols);
#endif
}
// # Spec
// # dev_sizes[] = { n-cols, n-rows, n-depth }
// # off[] = { x, y, z } (With respective to number of elements)
// # host_sizes[] = { x, y, z } (sizes with respective to number of elements)
void Utils::MemCpy3D(uint8_t *dev, uint8_t *host, size_t *off, 
        size_t *dev_sizes, size_t *host_sizes, 
        size_t elem_size, bool host_2_dev)
{
    size_t host_row_pitch = elem_size * host_sizes[0];
    size_t host_slice_pitch   = host_sizes[1] * host_row_pitch;
    size_t dev_row_pitch = elem_size * dev_sizes[0];
    size_t dev_slice_pitch = dev_sizes[1] * dev_row_pitch;
    uint8_t *host_start = host + off[0]*elem_size + off[1] * host_row_pitch + off[2] * host_slice_pitch;
    size_t dev_off[3] = {  0, 0, 0 };
    uint8_t *dev_start = dev + dev_off[0] * elem_size + dev_off[1] * dev_row_pitch + dev_off[2] * dev_slice_pitch;
    //printf("host_row_pitch:%d host_slice_pitch:%d\n", host_row_pitch, host_slice_pitch);
    //printf("DEV SIZE:(%d, %d, %d)\n", dev_sizes[0], dev_sizes[1], dev_sizes[2]);
    //printf("HOST SIZE:(%d, %d, %d)\n", host_sizes[0], host_sizes[1], host_sizes[2]);
    //printf("OFF:(%d, %d, %d) ELEM_SIZE:%d\n", off[0], off[1], off[2], elem_size);
    //printf("Host:%p Dest:%p\n", host_start, dev_start);
    for(size_t i=0; i<dev_sizes[2]; i++) {
        uint8_t *z_host = host_start + i * host_slice_pitch;
        uint8_t *z_dev = dev_start + i * dev_slice_pitch;
        for(size_t j=0; j<dev_sizes[1]; j++) {
            uint8_t *y_host = z_host + j * host_row_pitch;
            uint8_t *d_dev = z_dev + j * dev_row_pitch;
            if (host_2_dev) {
                //printf("0(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                memcpy(d_dev, y_host, dev_sizes[0]*elem_size);
            }
            else {
                //printf("1(%d:%d) Host:%p Dest:%p Size:%d\n", i, j, y_host, d_dev, dev_sizes[0]);
                memcpy(y_host, d_dev, dev_sizes[0]*elem_size);
            }
        }
    }
}

void Utils::Logo(bool color) {
  if (color) {
    srand(time(NULL));
    char str[12];
    sprintf(str, "\x1b[%d;3%dm", rand() & 1, rand() % 8 + 1);
    printf("%s", str);
  }
  printf("     ██╗     ██╗██████╗ ██╗███████╗        ██╗  \n");
  printf("     ╚██╗    ██║██╔══██╗██║██╔════╝        ╚██╗ \n");
  printf("█████╗╚██╗   ██║██████╔╝██║███████╗   █████╗╚██╗\n");
  printf("╚════╝██╔╝   ██║██╔══██╗██║╚════██║   ╚════╝██╔╝\n");
  printf("     ██╔╝    ██║██║  ██║██║███████║        ██╔╝ \n");
  printf("     ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚══════╝        ╚═╝  \n");
  if (color) {
    printf(RESET);
    fprintf(stderr, RESET);
  }
}

template <typename T>
void Utils::Fill(T* array_ptr, size_t n_elements, T value) {
    if (array_ptr == nullptr) {
        std::cerr << "Null pointer error: cannot fill a null array." << std::endl;
        return;
    }
    std::fill(array_ptr, array_ptr + n_elements, value);  // Fill using pointer arithmetic
}

void Utils::Fill(void *array_ptr, size_t n_elements, int element_type, ResetData &reset) {
    switch(element_type) {
        case iris_float:  Fill<float   >((float*)array_ptr, n_elements, reset.value_.f32); break;
        case iris_double: Fill<double  >((double*)array_ptr, n_elements, reset.value_.f64); break;
        case iris_uint64: Fill<uint64_t>((uint64_t*)array_ptr, n_elements, reset.value_.u64); break;
        case iris_uint32: Fill<uint32_t>((uint32_t*)array_ptr, n_elements, reset.value_.u32); break;
        case iris_uint16: Fill<uint16_t>((uint16_t*)array_ptr, n_elements, reset.value_.u16); break;
        case iris_uint8:  Fill<uint8_t >((uint8_t*)array_ptr, n_elements, reset.value_.u8); break;
        case iris_int64:  Fill<int64_t >((int64_t*)array_ptr, n_elements, reset.value_.i64); break;
        case iris_int32:  Fill<int32_t >((int32_t*)array_ptr, n_elements, reset.value_.i32); break;
        case iris_int16:  Fill<int16_t >((int16_t*)array_ptr, n_elements, reset.value_.i16); break;
        case iris_int8:   Fill<int8_t  >((int8_t*)array_ptr, n_elements, reset.value_.i8); break;
        default: break;
    }
}

template <typename T>
void Utils::ArithSequence(T* array_ptr, size_t n_elements, T start, T increment) {
    if (array_ptr == nullptr) {
        std::cerr << "Null pointer error: cannot fill a null array." << std::endl;
        return;
    }
    for (size_t i = 0; i < n_elements; ++i) {
        array_ptr[i] = start + i * increment;  // Fill with sequence
    }
}

void Utils::ArithSequence(void *array_ptr, size_t n_elements, int element_type, ResetData &reset) {
    switch(element_type) {
        case iris_float:  ArithSequence<float   >((float*)array_ptr, n_elements, reset.start_.f32, reset.step_.f32); break;
        case iris_double: ArithSequence<double  >((double*)array_ptr, n_elements, reset.start_.f64, reset.step_.f64); break;
        case iris_uint64: ArithSequence<uint64_t>((uint64_t*)array_ptr, n_elements, reset.start_.u64, reset.step_.u64); break;
        case iris_uint32: ArithSequence<uint32_t>((uint32_t*)array_ptr, n_elements, reset.start_.u32, reset.step_.u32); break;
        case iris_uint16: ArithSequence<uint16_t>((uint16_t*)array_ptr, n_elements, reset.start_.u16, reset.step_.u16); break;
        case iris_uint8:  ArithSequence<uint8_t >((uint8_t*)array_ptr, n_elements, reset.start_.u8, reset.step_.u8); break;
        case iris_int64:  ArithSequence<int64_t >((int64_t*)array_ptr, n_elements, reset.start_.i64, reset.step_.i64); break;
        case iris_int32:  ArithSequence<int32_t >((int32_t*)array_ptr, n_elements, reset.start_.i32, reset.step_.i32); break;
        case iris_int16:  ArithSequence<int16_t >((int16_t*)array_ptr, n_elements, reset.start_.i16, reset.step_.i16); break;
        case iris_int8:   ArithSequence<int8_t  >((int8_t*)array_ptr, n_elements, reset.start_.i8, reset.step_.i8); break;
        default: break;
    }
}

template <typename T>
void Utils::GeometricSequence(T* array_ptr, size_t n_elements, T start, T ratio) {
    if (array_ptr == nullptr) {
        std::cerr << "Null pointer error: cannot fill a null array." << std::endl;
        return;
    }
    for (size_t i = 0; i < n_elements; ++i) {
        // Integer types: use loop-based multiplication
        T value = start;
        for (size_t j = 0; j < i; ++j) {
            value *= ratio;
        }
        array_ptr[i] = value;
    }
}

template <>
void Utils::GeometricSequence<float>(float* array_ptr, size_t n_elements, float start, float ratio) {
    if (array_ptr == nullptr) {
        std::cerr << "Null pointer error: cannot fill a null array." << std::endl;
        return;
    }
    for (size_t i = 0; i < n_elements; ++i) {
        array_ptr[i] = start * powf(ratio, i);  // Fill with sequence
    }
}

template <>
void Utils::GeometricSequence<double>(double* array_ptr, size_t n_elements, double start, double ratio) {
    if (array_ptr == nullptr) {
        std::cerr << "Null pointer error: cannot fill a null array." << std::endl;
        return;
    }
    for (size_t i = 0; i < n_elements; ++i) {
        array_ptr[i] = start * pow(ratio, i);  // Fill with sequence
    }
}

void Utils::GeometricSequence(void *array_ptr, size_t n_elements, int element_type, ResetData &reset) {
    switch(element_type) {
        case iris_float:  GeometricSequence<float   >((float*)array_ptr, n_elements, reset.start_.f32, reset.step_.f32); break;
        case iris_double: GeometricSequence<double  >((double*)array_ptr, n_elements, reset.start_.f64, reset.step_.f64); break;
        case iris_uint64: GeometricSequence<uint64_t>((uint64_t*)array_ptr, n_elements, reset.start_.u64, reset.step_.u64); break;
        case iris_uint32: GeometricSequence<uint32_t>((uint32_t*)array_ptr, n_elements, reset.start_.u32, reset.step_.u32); break;
        case iris_uint16: GeometricSequence<uint16_t>((uint16_t*)array_ptr, n_elements, reset.start_.u16, reset.step_.u16); break;
        case iris_uint8:  GeometricSequence<uint8_t >((uint8_t*)array_ptr, n_elements, reset.start_.u8, reset.step_.u8); break;
        case iris_int64:  GeometricSequence<int64_t >((int64_t*)array_ptr, n_elements, reset.start_.i64, reset.step_.i64); break;
        case iris_int32:  GeometricSequence<int32_t >((int32_t*)array_ptr, n_elements, reset.start_.i32, reset.step_.i32); break;
        case iris_int16:  GeometricSequence<int16_t >((int16_t*)array_ptr, n_elements, reset.start_.i16, reset.step_.i16); break;
        case iris_int8:   GeometricSequence<int8_t  >((int8_t*)array_ptr, n_elements, reset.start_.i8, reset.step_.i8); break;
        default: break;
    }
}

template <typename T>
void Utils::RandomUniformSeq(T *arr, long long seed, size_t size, T p1, T p2) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<> dist(p1, p2);
    //#pragma omp parallel for
    for(size_t i=0; i<size; i++) {
        arr[i] = dist(generator);
        //printf("Arr: %d %f\n", i, (float)arr[i]);
    }
}

void Utils::RandomUniformSeq(void *array_ptr, size_t size, int element_type, ResetData &reset) {
#define RU_SEQ(IT, T, M)   case IT: RandomUniformSeq<T>((T*)array_ptr, reset.seed_, size, reset.p1_.M, reset.p2_.M); break;
    switch(element_type) {
        RU_SEQ(iris_float, float, f32);
        RU_SEQ(iris_double, double, f64);
        RU_SEQ(iris_uint64, uint64_t, u64);
        RU_SEQ(iris_uint32, uint32_t, u32);
        RU_SEQ(iris_uint16, uint16_t, u16);
        RU_SEQ(iris_uint8,  uint8_t,  u8);
        RU_SEQ(iris_int64,  int64_t,  i64);
        RU_SEQ(iris_int32,  int32_t,  i32);
        RU_SEQ(iris_int16,  int16_t,  i16);
        RU_SEQ(iris_int8,   int8_t,   i8);
        default: break;
    }
}

template <typename T>
void Utils::RandomNormalSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    std::mt19937 generator(seed);
    std::normal_distribution<> dist(p1, p2);
    //#pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = dist(generator);
    }
}

template <typename T>
void Utils::RandomLogNormalSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    std::mt19937 generator(seed);
    std::lognormal_distribution<> dist(p1, p2);
    //#pragma omp parallel for
    for(int i=0; i<size; i++) {
        arr[i] = dist(generator);
    }
}

template <typename T>
void Utils::RandomUniformSobolSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    fprintf(stderr, "[Error] Undefined openmp function: %s\n", __func__);
}

template <typename T>
void Utils::RandomNormalSobolSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    fprintf(stderr, "[Error] Undefined openmp function: %s\n", __func__);
}

template <typename T>
void Utils::RandomLogNormalSobolSeq(T *arr, unsigned long long seed, size_t size, T p1, T p2, void *stream) {
    fprintf(stderr, "[Error] Undefined openmp function: %s\n", __func__);
}

int Utils::ReadFile(char* path, char** string, size_t* len) {
  int fd = open((const char*) path, O_RDONLY);
  if (fd == -1) {
    *len = 0UL;
    return IRIS_ERROR;
  }
  off_t s = lseek(fd, 0, SEEK_END);
  *string = (char*) calloc(sizeof(char),s+1);
  *len = s;
  lseek(fd, 0, SEEK_SET);
  ssize_t r = read(fd, *string, s);
  if (r != s) {
    _error("read[%zd] vs [%lu]", r, s);
    return IRIS_ERROR;
  }
  close(fd);
  return IRIS_SUCCESS;
}

int Utils::Mkdir(char* path) {
  struct stat st;
  if (stat(path, &st) == -1) {
    if (mkdir(path, 0700) == -1) {
      return IRIS_ERROR;
    }
  }
  return IRIS_SUCCESS;
}

bool Utils::Exist(char* path) {
  struct stat st;
  return stat(path, &st) != -1;
}

long Utils::Mtime(char* path) {
  struct stat st;
  if (stat(path, &st) == -1) return -1;
  return st.st_mtime;
}

void Utils::ReadSet(std::set<std::string> & out, const char *data)
{
    const char* delim = " :;.,";
    std::string data_str = std::string(data);
    std::transform(data_str.begin(), data_str.end(), data_str.begin(), ::tolower);
    char* rest = (char *)data_str.c_str();
    char* a = NULL;
    while ((a = strtok_r(rest, delim, &rest))) {
        out.insert(a);
    }
}

void Utils::ReadVector(std::vector<std::string> & out, const char *data)
{
    const char* delim = " :;.,";
    std::string data_str = std::string(data);
    std::transform(data_str.begin(), data_str.end(), data_str.begin(), ::tolower);
    char* rest = (char *)data_str.c_str();
    char* a = NULL;
    while ((a = strtok_r(rest, delim, &rest))) {
        out.push_back(a);
    }
}
void Utils::ReadMap(std::map<std::string, int> & in_map, std::map<int, bool> & out_map, const char *data)
{
    const char* delim = " :;.,";
    std::string data_str = std::string(data);
    std::transform(data_str.begin(), data_str.end(), data_str.begin(), ::tolower);
    char* rest = (char *)data_str.c_str();
    char* a = NULL;
    while ((a = strtok_r(rest, delim, &rest))) {
        if (in_map.find(a) != in_map.end()) {
            out_map[in_map[a]] = true;
        }
    }
}
void Utils::Datetime(char* str) {
  time_t t = time(NULL);
  struct tm* tm = localtime(&t);
  strftime(str, 256, "%Y%m%d%H%M%S", tm);
}

int Utils::CPUCoresCount() {
    unsigned int num_cores = std::thread::hardware_concurrency();
    return num_cores;
}
void Utils::SetThreadAffinity(unsigned int core_id) {
    unsigned int num_cores = std::thread::hardware_concurrency();
    unsigned int lcore_id = core_id;
    if (lcore_id >= num_cores) 
        lcore_id = lcore_id % num_cores;
#ifdef _WIN32
    DWORD_PTR mask = 1ULL << lcore_id;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else //_WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(lcore_id, &cpuset);

#if 1
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#else
    pthread_t current_thread = pthread_self();
    int s = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    printf("Device:%u Setting affinity of thread to core id:%u\n", core_id, lcore_id);
    if (s != 0)
        _error("CPU affinity set to %u is failed with pthread_setaffinity_np", core_id);
#endif
#endif // _WIN32
}

} /* namespace rt */
} /* namespace iris */
