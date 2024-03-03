#include <iris/iris.h>
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
#include <execinfo.h>
#include <cstdlib>
#include <regex>
#include <cxxabi.h>
#include <thread>

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

    pthread_t current_thread = pthread_self();
    int s = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    printf("Device:%u Setting affinity of thread to core id:%u\n", core_id, lcore_id);
    if (s != 0)
        _error("CPU affinity set to %u is failed with pthread_setaffinity_np", core_id);
#endif // _WIN32
}

} /* namespace rt */
} /* namespace iris */
