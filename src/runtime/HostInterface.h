#pragma once

#include "Loader.h"
#include "Kernel.h"
#include "Debug.h"

#include <string>
#include <map>
#ifndef DISABLE_DYNAMIC_LINKING
#define REGISTER_HOST_WRAPPER(FN, SYM)  LoadFunction(#FN, #SYM); 
#else
#define REGISTER_HOST_WRAPPER(FN, SYM)  LoadFunctionStatic(#FN, CLinkage::SYM); 
#endif

#ifdef ENABLE_FFI
#include "ffi.h"
#define HostInterfaceClass FFIHostInterfaceLoader
#else
#define HostInterfaceClass BoilerPlateHostInterfaceLoader
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
        class Command;
        class Kernel;
        // Note: Do not use these base and derived classes to hold some state other than 
        // function pointer. The loaders are shared across same homogeneous devices
        class HostInterfaceLoader : public Loader {
            public:
                HostInterfaceLoader(string kernel_env);
                HostInterfaceLoader(string kernel_env, string suffix);
                ~HostInterfaceLoader() { }
                virtual const char* library();
#ifdef DISABLE_DYNAMIC_LINKING
                virtual bool IsFunctionExists(const char *kernel_name);
                virtual void *GetFunctionPtr(const char *kernel_name);
#endif
                virtual int LoadFunctions(){ Loader::LoadFunctions(); return IRIS_SUCCESS; }
                virtual void finalize(int dev);
                virtual void init(int dev);
                virtual int launch_init(int model, int *dev_ptr, void *stream, void *param_mem, Command *cmd_kernel) { return IRIS_SUCCESS; }
                virtual int setarg(void *param_mem, int index, size_t size, void *value) { return IRIS_ERROR; }
                virtual int setmem(void *param_mem, int kindex, void *mem) { return IRIS_ERROR; }
                virtual int host_launch(void **stream, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws) { return IRIS_ERROR; }
                virtual int host_kernel(void *param_mem, const char *kname) { return IRIS_ERROR; }
                int SetKernelPtr(void *obj, const char *kernel_name) { return IRIS_ERROR; }
                void LoadFunction(const char *func_name, const char *symbol);
                void LoadFunctionStatic(const char *func_name, void *symbol);
            protected:
                map<string, void **> sym_map_fn_;
                int (*iris_host_init)();
                int (*iris_host_init_handles)(int devno);
                int (*iris_host_finalize_handles)(int devno);
                int (*iris_host_finalize)();
                int (*iris_host_launch)(int dim, size_t *off, size_t *bws);
                int (*iris_ffi_launch)();
                int (*iris_host_launch_with_obj)(void *stream, void *obj, int devno, int dim, size_t *off, size_t *bws);
            private:
                string kernel_env_;
        };
        class BoilerPlateHostInterfaceLoader : public HostInterfaceLoader {
            public:
                BoilerPlateHostInterfaceLoader(string kernel_env);
                ~BoilerPlateHostInterfaceLoader() { }
                virtual const char* library() { return HostInterfaceLoader::library(); }
                int LoadFunctions();
                int host_kernel(void *param_mem, const char *kname);
                int SetKernelPtr(void *obj, const char *kernel_name);
                int launch_init(int model, int *dev_ptr, void *stream, void *param_mem, Command *cmd);
                int host_launch(void **stream, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws);
                int setarg(void *param_mem, int kindex, size_t size, void *value);
                int setmem(void *param_mem, int kindex, void *mem);
            private:
                int (*iris_host_kernel)(const char* name);
                int (*iris_host_setarg)(int idx, size_t size, void* value);
                int (*iris_host_setmem)(int idx, void* mem);
                int (*iris_host_kernel_with_obj)(void *obj, const char* name);
                int (*iris_host_setarg_with_obj)(void *obj, int idx, size_t size, void* value);
                int (*iris_host_setmem_with_obj)(void *obj, int idx, void* mem);
                void (*iris_set_kernel_ptr_with_obj)(void *obj, __iris_kernel_ptr ptr);
                c_string_array (*iris_get_kernel_names)();
        };
#ifdef ENABLE_FFI
#include "ffi.h"
        enum FFICallType {
            HOST_LAUNCH_WITH_STREAM_DEV,
            HOST_LAUNCH_BARE,
            HOST_LAUNCH_WITH_FFI_OBJ,
            HOST_LAUNCH_FFI_BARE
        };
        class KernelFFI {
            public:
                KernelFFI( ) {
                    index_ = 0;
                    args_ = NULL;
                    values_ = NULL;
                    values_ptr_ = NULL;
                    args_capacity_ = 0;
                }
                ~KernelFFI( ) {
                    if (args_ != NULL) free(args_);
                    if (values_ != NULL) free(values_);
                    if (values_ptr_ != NULL) free(values_ptr_);
                }
                void init(int nargs) {
                    args_ = NULL;
                    values_ = NULL;
                    values_ptr_ = NULL;
                    args_capacity_ = nargs+6;
                    if (args_ == NULL) 
                        args_ = (ffi_type **) malloc(
                                sizeof(ffi_type *)*args_capacity_);
                    if (values_ == NULL)
                        values_ = (void **) malloc(
                                sizeof(void *)*args_capacity_);
                    if (values_ptr_ == NULL)
                        values_ptr_ = (void **) malloc(
                                sizeof(void *)*args_capacity_);
                    index_  = 0;
                }
                void set_kernel(Kernel *kernel) { kernel_ = kernel; }
                void set_arg_type(ffi_type *arg_type) {
                    assert(index_ < args_capacity_);
                    args_[index_] = arg_type;
                }
                void set_value(void *value) { 
                    assert(index_ < args_capacity_);
                    values_[index_] = value; 
                    //printf("         Values ptr:%p\n", values_+index_); 
                }
                void set_value_ptr(void *value) { 
                    values_ptr_[index_] = value; 
                    values_[index_] = &values_ptr_[index_]; 
                    //printf("         Values mem:%p mem_ptr:%p\n", values_ptr_[index_], values_[index_]); 
                }
                void set_host_launch_type(FFICallType type) { type_ = type; }
                FFICallType host_launch_type() { return type_; }
                void set_fn_ptr(__iris_kernel_ptr ptr) { fn_ptr_ = ptr; }
                void set_iris_args(KernelArg *iris_args) { iris_args_ = iris_args; }
                KernelArg *get_iris_arg(int kindex) { return iris_args_ + kindex; }
                void **values() { return values_; }
                int top() { return index_; }
                void increment() { index_++; }
                void add_epilog(size_t off, size_t ndr) {
                    param_idx_ = off;
                    param_size_ = ndr;
                    set_arg_type(&ffi_type_uint64);
                    set_value(&param_idx_);
                    increment();
                    set_arg_type(&ffi_type_uint64);
                    set_value(&param_size_);
                    increment();
                }
                void add_epilog(size_t *off, size_t *bws) {
                    set_arg_type(&ffi_type_pointer);
                    set_value(off);
                    increment();
                    set_arg_type(&ffi_type_pointer);
                    set_value(bws);
                    increment();
                }
                void add_stream(void *stream) {
                    set_arg_type(&ffi_type_pointer);
                    set_value(stream);
                    increment();
                }
                void add_device(int *dev_ptr) {
                    set_arg_type(&ffi_type_sint);
                    set_value(dev_ptr);
                    increment();
                }
                ffi_cif *cif() { return &cif_; }
                ffi_type **args() { return args_; }
                __iris_kernel_ptr fn_ptr() { return fn_ptr_; }
                Kernel *kernel() { return kernel_; }
            private:
                int args_capacity_;
                Kernel *kernel_;
                __iris_kernel_ptr fn_ptr_;
                KernelArg *iris_args_;
                ffi_type **args_;
                void **values_;
                void **values_ptr_;
                ffi_cif cif_;
                size_t param_size_;
                size_t param_idx_;
                FFICallType type_;
                int index_;
        };
        class FFIHostInterfaceLoader : public HostInterfaceLoader {
            public:
                FFIHostInterfaceLoader(string kernel_env);
                ~FFIHostInterfaceLoader() { }
                virtual const char* library() { return HostInterfaceLoader::library(); }
                int LoadFunctions();
                int host_kernel(void *param_mem, const char *kname);
                void set_kernel_ffi(void *param_mem, KernelFFI *ffi_data);
                KernelFFI *get_kernel_ffi(void *param_mem);
                int launch_init(int model, int *dev_ptr, void *stream, void *param_mem, Command *cmd);
                int SetKernelPtr(void *obj, const char *kernel_name);
                int host_launch(void **stream, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws);
                int setarg(void *param_mem, int kindex, size_t size, void *value);
                int setmem(void *param_mem, int kindex, void *mem);
            private:
                ffi_cif cif_;
        };
#endif
    }
}
