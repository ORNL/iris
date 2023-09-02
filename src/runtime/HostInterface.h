#pragma once

#include "Loader.h"

#include <string>
#include <map>
#ifndef ENABLE_STATIC_LINKING
#define REGISTER_HOST_WRAPPER(FN, SYM)  LoadFunction(#FN, #SYM); 
#else
#define REGISTER_HOST_WRAPPER(FN, SYM)  LoadFunctionStatic(#FN, CLinkage::SYM); 
#endif
using namespace std;
namespace CLinkage {
    extern "C" {
        void iris_set_kernel_ptr_with_obj(void *obj, __iris_kernel_ptr ptr);
        c_string_array iris_get_kernel_names();
    }
}
namespace iris {
    namespace rt {
        class HostInterfaceLoader : public Loader {
            public:
                HostInterfaceLoader(string kernel_env) : Loader(), kernel_env_(kernel_env), dev_(0) { }
                HostInterfaceLoader(int dev, string kernel_env) : Loader(), kernel_env_(kernel_env), dev_(dev) { }
                ~HostInterfaceLoader() { }
                const char* library();
                virtual int LoadFunctions(){ return IRIS_ERROR; }
                virtual void init() {}
                virtual void finalize() {}
                virtual void setarg(void *param_mem, int index, size_t size, void *value) {}
                virtual void setmem(void *param, int index, void *mem) {}
                virtual int host_launch(void *stream, const char *kname, void *param_mem, int dim, size_t *off, size_t *gws) { return IRIS_ERROR; }
                virtual int host_kernel(void *param_mem, const char *kname) { return IRIS_ERROR; }
                int SetKernelPtr(void *obj, const char *kernel_name) { return IRIS_ERROR; }
                virtual void LoadFunction(const char *func_name, const char *symbol) {}
                virtual void LoadFunctionStatic(const char *func_name, void *symbol) {}
                void set_dev(int dev) { dev_ = dev; }
                int dev() { return dev_; }
            private:
                string kernel_env_;
                int dev_;

        };
        class BoilerPlateHostInterfaceLoader : public HostInterfaceLoader {
            public:
                BoilerPlateHostInterfaceLoader(string kernel_env);
                ~BoilerPlateHostInterfaceLoader() { }
                int LoadFunctions();
                void LoadFunction(const char *func_name, const char *symbol);
                void LoadFunctionStatic(const char *func_name, void *symbol);
                int host_kernel(void *param_mem, const char *kname);
                int SetKernelPtr(void *obj, const char *kernel_name);
                int host_launch(void *stream, const char *kname, void *param_mem, int dim, size_t *off, size_t *gws);
                void setarg(void *param_mem, int kindex, size_t size, void *value);
                void setmem(void *param_mem, int kindex, void *dev_ptr);
                void finalize();
                void init();
            private:
                map<string, void **> sym_map_fn_;
                int (*iris_host_init)();
                int (*iris_host_init_handles)(int devno);
                int (*iris_host_finalize_handles)(int devno);
                int (*iris_host_finalize)();
                int (*iris_host_kernel)(const char* name);
                int (*iris_host_setarg)(int idx, size_t size, void* value);
                int (*iris_host_setmem)(int idx, void* mem);
                int (*iris_host_launch)(int dim, size_t off, size_t gws);
                int (*iris_host_kernel_with_obj)(void *obj, const char* name);
                int (*iris_host_setarg_with_obj)(void *obj, int idx, size_t size, void* value);
                int (*iris_host_setmem_with_obj)(void *obj, int idx, void* mem);
                int (*iris_host_launch_with_obj)(void *obj, int devno, int dim, size_t off, size_t gws);
                void (*iris_set_kernel_ptr_with_obj)(void *obj, __iris_kernel_ptr ptr);
                c_string_array (*iris_get_kernel_names)();
        };
    }
}
