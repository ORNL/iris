#ifndef IRIS_SRC_RT_DEBUG_H
#define IRIS_SRC_RT_DEBUG_H

#include <assert.h>
#include <stdio.h>
#include <string.h>

#ifndef TRACE_DISABLE
#define _TRACE_ENABLE
#endif //TRACE_DISABLE
#ifndef CHECK_DISABLE
#define _CHECK_ENABLE
#endif //CHECK_DISABLE
#ifndef DEBUG_DISABLE
#define _DEBUG_ENABLE
#endif //DEBUG_DISABLE
#ifndef INFO_DISABLE
#define _INFO_ENABLE
#endif //INFO_DISABLE
#ifndef TODO_DISABLE
#define _TODO_ENABLE
#endif //TODO_DISABLE
#ifndef WARNING_DISABLE
#define _WARNING_ENABLE
#endif //WARNING_DISABLE

#define _PRINTF_ENABLE
#define FFLUSH(X)    //fflush(X)
#define _ERROR_ENABLE
#define _CLERROR_ENABLE
#define _CUERROR_ENABLE
#define _HIPERROR_ENABLE
#define _ZEERROR_ENABLE
#define _HIPWARNING_ENABLE
#define _CUWARNING_ENABLE
#define _CLWARNING_ENABLE

#define _COLOR_DEBUG

#define ASSERT(X)   assert(X)
#ifdef _COLOR_DEBUG
#define RED     "\033[22;31m"
#define GREEN   "\033[22;32m"
#define YELLOW  "\033[22;33m"
#define BLUE    "\033[22;34m"
#define PURPLE  "\033[22;35m"
#define CYAN    "\033[22;36m"
#define GRAY    "\033[22;37m"
#define BRED    "\033[1;31m"
#define BGREEN  "\033[1;32m"
#define BYELLOW "\033[1;33m"
#define BBLUE   "\033[1;34m"
#define BPURPLE "\033[1;35m"
#define BCYAN   "\033[1;36m"
#define BGRAY   "\033[1;37m"
#define _RED    "\033[22;41m" BGRAY
#define _GREEN  "\033[22;42m" BGRAY
#define _YELLOW "\033[22;43m" BGRAY
#define _BLUE   "\033[22;44m" BGRAY
#define _PURPLE "\033[22;45m" BGRAY
#define _CYAN   "\033[22;46m" BGRAY
#define _GRAY   "\033[22;47m"
#define RESET   "\x1b[m"
#else
#define RED
#define GREEN
#define YELLOW
#define BLUE
#define PURPLE
#define CYAN
#define GRAY
#define BRED
#define BGREEN
#define BYELLOW
#define BBLUE
#define BPURPLE
#define BCYAN
#define BGRAY
#define _RED
#define _GREEN
#define _YELLOW
#define _BLUE
#define _PURPLE
#define _CYAN
#define _GRAY
#define RESET
#endif

#define CHECK_O   "\u2714 "
#define CHECK_X   "\u2716 "

namespace iris {
namespace rt {

extern char iris_log_prefix_[];

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

//#define  _printf(fmt, ...) do { printf( GREEN "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)

//#define _DEBUG2_ENABLE
#ifdef _DEBUG2_ENABLE
#define  _debug2(fmt, ...) do { printf( GREEN "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _debug2(fmt, ...) do { } while (0)
#endif

//#define _DEBUG2_EVENT_PROF
#ifdef _DEBUG2_EVENT_PROF
#define  _event_prof_debug(fmt, ...) do { printf( "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _event_prof_debug(fmt, ...) do { } while (0)
#endif


//#define _DEBUG_EVENT
#ifdef _DEBUG_EVENT
#define EVENT_DEBUG(X)   X
#define  _event_debug(fmt, ...) do { printf( "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define EVENT_DEBUG(X)   
#define  _event_debug(fmt, ...) do { } while (0)
#endif

#ifdef _TRACE_ENABLE
#define  _trace(fmt, ...) do { printf( BLUE "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define  _trace_debug(fmt, ...) do { printf( BRED "[T] Manual Debug %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define __trace(fmt, ...) do { printf(_BLUE "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
//#define  _printf(fmt, ...) do { printf( GREEN "[T] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _trace(fmt, ...) do { } while (0)
#define  __trace(fmt, ...) do { } while (0)
#define  _trace_debug(fmt, ...) do { } while (0)
//#define  _printf(fmt, ...) do { } while (0)
#endif

#ifdef _CHECK_ENABLE
#define  _check() do { printf( PURPLE "[C] %s [%s:%d:%s]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__); FFLUSH(stdout); } while (0)
#define __check() do { printf(_PURPLE "[C] %s [%s:%d:%s]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__); FFLUSH(stdout); } while (0)
#else
#define  _check() do { } while (0)
#define __check() do { } while (0)
#endif

#ifdef _DEBUG_ENABLE
#define  _debug(fmt, ...) do { printf( CYAN "[D] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define __debug(fmt, ...) do { printf(_CYAN "[D] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _debug(fmt, ...) do { } while (0)
#define __debug(fmt, ...) do { } while (0)
#endif

#ifdef _PRINTF_ENABLE
#define  _printf(fmt, ...) do { printf( YELLOW "[I] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define __printf(fmt, ...) do { printf(_YELLOW "[I] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _printf(fmt, ...) do { } while (0)
#define __printf(fmt, ...) do { } while (0)
#endif

#ifdef _INFO_ENABLE
#define  _info(fmt, ...) do { printf( YELLOW "[I] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define __info(fmt, ...) do { printf(_YELLOW "[I] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _info(fmt, ...) do { } while (0)
#define __info(fmt, ...) do { } while (0)
#endif

#ifdef _ERROR_ENABLE
#define  _error(fmt, ...) do { printf( RED "[E] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define ___error(fmt, ...) do { printf(_RED "[E] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0) // MacOS
#else
#define  _error(fmt, ...) do { } while (0)
#define ___error(fmt, ...) do { } while (0) // MacOS
#endif

#ifdef _WARNING_ENABLE
#define  _warning(fmt, ...) do { printf( CYAN "[W] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define ___warning(fmt, ...) do { printf(_CYAN "[W] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0) // MacOS
#else
#define  _warning(fmt, ...) do { } while (0)
#define ___warning(fmt, ...) do { } while (0) // MacOS
#endif

#ifdef _TODO_ENABLE
#define  _todo(fmt, ...) do { printf( GREEN "[TODO] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#define __todo(fmt, ...) do { printf(_GREEN "[TODO] %s [%s:%d:%s] " fmt RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, ##__VA_ARGS__); FFLUSH(stdout); } while (0)
#else
#define  _todo(fmt, ...) do { } while (0)
#define __todo(fmt, ...) do { } while (0)
#endif

#ifdef _CLERROR_ENABLE
#define  _clerror(err) do { if (err != CL_SUCCESS) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __clerror(err) do { if (err != CL_SUCCESS) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _clerror(err) do { } while (0)
#define __clerror(err) do { } while (0)
#endif

#ifdef _CUERROR_ENABLE
#define  _cuerror(err) do { if (err != CUDA_SUCCESS) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __cuerror(err) do { if (err != CUDA_SUCCESS) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _cuerror(err) do { } while (0)
#define __cuerror(err) do { } while (0)
#endif

#ifdef _HIPERROR_ENABLE
#define  _hiperror(err) do { if (err != hipSuccess) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __hiperror(err) do { if (err != hipSuccess) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _hiperror(err) do { } while (0)
#define __hiperror(err) do { } while (0)
#endif

#ifdef _CLWARNING_ENABLE
#define  _clwarning(err) do { if (err != CL_SUCCESS) { printf( RED "[W] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __clwarning(err) do { if (err != CL_SUCCESS) { printf(_RED "[W] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _clwarning(err) do { } while (0)
#define __clwarning(err) do { } while (0)
#endif

#ifdef _CUWARNING_ENABLE
#define  _cuwarning(err) do { if (err != CUDA_SUCCESS) { printf( RED "[W] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __cuwarning(err) do { if (err != CUDA_SUCCESS) { printf(_RED "[W] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _cuwarning(err) do { } while (0)
#define __cuwarning(err) do { } while (0)
#endif

#ifdef _HIPWARNING_ENABLE
#define  _hipwarning(err) do { if (err != hipSuccess) { printf( RED "[W] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __hipwarning(err) do { if (err != hipSuccess) { printf(_RED "[W] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _hipwarning(err) do { } while (0)
#define __hipwarning(err) do { } while (0)
#endif


#ifdef _ZEERROR_ENABLE
#define  _zeerror(err) do { if (err != ZE_RESULT_SUCCESS) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#define __zeerror(err) do { if (err != ZE_RESULT_SUCCESS) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", iris_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); FFLUSH(stdout); } } while (0)
#else
#define  _zeerror(err) do { } while (0)
#define __zeerror(err) do { } while (0)
#endif

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_DEBUG_H */
