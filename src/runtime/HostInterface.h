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
typedef struct _jl_value_t jl_value_t;
typedef jl_value_t jl_function_t;
void *jl_unbox_voidpointer(jl_value_t *v);
void jl_gc_add_finalizer(jl_value_t *v, jl_function_t *f);
void jl_init(void);
void jl_atexit_hook(int);
int jl_is_initialized(void);

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
                virtual ~HostInterfaceLoader() { }
                virtual const char* library();
#ifdef DISABLE_DYNAMIC_LINKING
                virtual bool IsFunctionExists(const char *kernel_name);
                virtual void *GetFunctionPtr(const char *kernel_name);
#endif
                virtual int LoadFunctions(){ Loader::LoadFunctions(); return IRIS_SUCCESS; }
                virtual void finalize(int dev);
                virtual void init(int dev);
                virtual int launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd_kernel) { return IRIS_SUCCESS; }
                virtual int setarg(void *param_mem, int index, size_t size, void *value) { return IRIS_ERROR; }
                virtual int setmem(void *param_mem, int kindex, void *mem, size_t size) { return IRIS_ERROR; }
                virtual int setmem(void *param_mem, BaseMem *mem, int kindex, void *mem_ptr, size_t size) { return IRIS_ERROR; }
                static uint32_t get_encoded_stream_device(int stream_index, int nstreams, int device);
                static uint32_t update_dev_info_stream_index(int dev_info, int stream_index);
                static uint32_t update_dev_info_nstreams(int dev_info, int nstreams);
                static uint32_t update_dev_info_devno(int dev_info, int devno);
                virtual int host_launch(void **stream, int stream_index, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws) { return IRIS_ERROR; }
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
                int (*iris_host_launch_with_obj)(void *stream, void *obj, uint32_t devno, int dim, size_t *off, size_t *bws);
                string & kernel_env() { return kernel_env_; }
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
                int launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd);
                int host_launch(void **stream, int stream_index, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws);
                int setarg(void *param_mem, int kindex, size_t size, void *value);
                int setmem(void *param_mem, int kindex, void *mem, size_t size);
            private:
                int (*iris_host_kernel)(const char* name);
                int (*iris_host_setarg)(int idx, size_t size, void* value);
                int (*iris_host_setmem)(int idx, void* mem);
                int (*iris_host_kernel_with_obj)(void *obj, const char* name);
                int (*iris_host_setarg_with_obj)(void *obj, int idx, size_t size, void* value);
                int (*iris_host_setmem_with_obj)(void *obj, int idx, void* mem, size_t size);
                void (*iris_set_kernel_ptr_with_obj)(void *obj, __iris_kernel_ptr ptr);
                c_string_array (*iris_get_kernel_names)();
        };
        class KernelJulia {
            public:
                KernelJulia( ) {
                    index_ = 0;
                    args_ = NULL;
                    values_ = NULL;
                    args_capacity_ = 0;
                    julia_kernel_type_ = iris_julia_native;
                    param_size_ = NULL;
                    param_dim_size_ = NULL;
                }
                void reset() {
                    index_ = 0;
                    args_ = NULL;
                    values_ = NULL;
                    julia_kernel_type_ = iris_julia_native;
                    args_capacity_ = 0;
                    param_size_ = NULL;
                    param_dim_size_ = NULL;
                }
                ~KernelJulia( ) {
                    if (args_ != NULL) free(args_);
                    if (values_ != NULL) free(values_);
                    if (param_size_ != NULL) free(param_size_);
                    if (param_dim_size_ != NULL) free(param_dim_size_);
                }
                void init(int nargs) {
                    args_ = NULL;
                    values_ = NULL;
                    args_capacity_ = nargs+6;
                    //printf("Setting args: %d param_size_:%p \n", args_capacity_, param_size_);
                    if (args_ == NULL) 
                        args_ = (int32_t *) malloc(
                                sizeof(int32_t *)*args_capacity_);
                    if (values_ == NULL)
                        values_ = (void **) malloc(
                                sizeof(void *)*args_capacity_);
                    if (param_size_ == NULL)
                        param_size_ = (size_t *) malloc(
                                sizeof(size_t) * args_capacity_);
                    if (param_dim_size_ == NULL)
                        param_dim_size_ = (size_t *)malloc(args_capacity_ * sizeof(size_t) * DMEM_MAX_DIM);
                    index_  = 0;
                }
                void set_kernel(Kernel *kernel) { kernel_ = kernel; }
                void set_arg_type(int32_t arg_type) {
                    assert(index_ < args_capacity_);
                    args_[index_] = arg_type;
                    //printf("Args:%p %d %d\n", args_, index_, arg_type);
                }
                void set_value(void *value) { 
                    assert(index_ < args_capacity_);
                    values_[index_] = value; 
                    //printf("         Values ptr:%p\n", values_+index_); 
                }
                void set_value_ptr(void *value) { 
                    values_[index_] = value;
                    //printf("         Values mem:%p mem_ptr:%p\n", values_ptr_[index_], values_[index_]); 
                }
                void set_dim_sizes(size_t *sizes, int dim) { 
                    ASSERT(dim < DMEM_MAX_DIM);
                    //printf("poor_dim_size[%d] %p dim:%d dev_size:%lu\n", index_, param_dim_size_ +index_*DMEM_MAX_DIM, dim, sizes[0]);
                    memcpy(param_dim_size_ + index_*DMEM_MAX_DIM, sizes, sizeof(size_t)*dim);
                }
                void set_size(size_t size) {
                    param_size_[index_] = size;
                }
                void set_fn_ptr(__iris_kernel_ptr ptr) { fn_ptr_ = ptr; }
                void set_iris_args(KernelArg *iris_args) { iris_args_ = iris_args; }
                KernelArg *get_iris_arg(int kindex) { return iris_args_ + kindex; }
                void **values() { return values_; }
                size_t *param_dim_size() { return param_dim_size_; }
                size_t *param_size() { return param_size_; }
                int top() { return index_; }
                void increment() { index_++; }
                void add_stream(void **stream) {
                    set_arg_type(iris_pointer);
                    streams_ = stream;
                    set_value(streams_);
                    increment();
                }
                void add_dev_info() {
                    set_arg_type(iris_int32);
                    set_value(&dev_info_);
                    increment();
                }
                int julia_kernel_type() { return julia_kernel_type_; }
                void set_julia_kernel_type(int type) { julia_kernel_type_ = type; }
                int32_t *args() { return args_; }
                __iris_kernel_ptr fn_ptr() { return fn_ptr_; }
                Kernel *kernel() { return kernel_; }
                void set_dev_info(int32_t stream_index, int nstreams, int devno) { dev_info_ = HostInterfaceLoader::get_encoded_stream_device(stream_index, nstreams, devno); }
                void set_nstream(int32_t nstreams) { dev_info_ = HostInterfaceLoader::update_dev_info_nstreams(dev_info_, nstreams); }
                void set_devno(int32_t devno) { dev_info_ = HostInterfaceLoader::update_dev_info_devno(dev_info_, devno); }
                void set_stream_index(int32_t stream_index) { dev_info_ = HostInterfaceLoader::update_dev_info_stream_index(dev_info_, stream_index); }
                //void set_kernel_name(const char *name) { kernel_name_ = name; }
                //const char *get_kernel_name() { return kernel_name_; }
            private:
                int args_capacity_;
                Kernel *kernel_;
                __iris_kernel_ptr fn_ptr_;
                KernelArg *iris_args_;
                int32_t *args_;
                void **values_;
                size_t *param_size_;
                size_t *param_dim_size_;
                size_t *param_idx_;
                void **streams_;
                uint32_t dev_info_;
                int julia_kernel_type_;
                int index_;
                //const char *kernel_name_;
        };
        class JuliaHostInterfaceLoader : public HostInterfaceLoader {
            public:
                JuliaHostInterfaceLoader(int model);
                ~JuliaHostInterfaceLoader() { }
                const char* library() { return "libjulia.so"; }
                void init(int dev);
                int LoadFunctions();
                int host_kernel(void *param_mem, const char *kname);
                void set_kernel_julia(void *param_mem, KernelJulia *julia_data);
                KernelJulia * get_kernel_julia(void *param_mem);
                int launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd);
                int SetKernelPtr(void *obj, const char *kernel_name);
                int host_launch(unsigned long task_id, void **stream, int stream_index, void *ctx, bool async, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws);
                int setarg(void *param_mem, int kindex, size_t size, void *value);
                int setmem(void *param_mem, BaseMem *mem, int kindex, void *mem_ptr, size_t size);
                int set_julia_kernel_type(void *param_mem, int type);
                void launch_julia_kernel(unsigned long task_id, int julia_kernel_type, int target, int32_t devno, void *ctx, bool async, int32_t stream_index, void **stream, int32_t nstreams, int32_t *args, void **values, size_t *param_size, size_t *param_dim_size, int32_t nparams, size_t *threads, size_t *blocks, int dim, const char *name)
                {
                    //printf("jl_is_initialized: %p jl_init:%p\n", jl_is_initialized, jl_init);
                    if (iris_get_julia_launch_func()!= NULL) {
                        //(*jl_init)();
                        julia_kernel_t kernel_julia_wrapper = iris_get_julia_launch_func();
                        //printf("Kernel ptr: %p %s args:%p values:%p\n", kernel_julia_wrapper, name, args, values);
                        //for(int i=0; i<nparams; i++) {
                        //    printf("Values:i:%d arg:%d:%d values:%p\n", i, args[i], args[i]>>16, values[i]);
                        //}
                        kernel_julia_wrapper(task_id, julia_kernel_type, target, devno, ctx, (int)async, stream_index, stream, nstreams, args, values, param_size, param_dim_size, nparams, threads, blocks, dim, name);
                        //printf("Result: %d\n", result);
                        //(*jl_atexit_hook)(0);
                    }
                }

                void *(*jl_unbox_voidpointer)(jl_value_t *v);
                void (*jl_gc_add_finalizer)(jl_value_t *v, jl_function_t *f);
                void (*jl_init)(void);
                void (*jl_atexit_hook)(int);
                int (*jl_is_initialized)(void);
            private:
                int target_;
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
                void add_epilog(size_t *off, size_t *ndr) {
                    param_idx_ = off;
                    param_size_ = ndr;
                    set_arg_type(&ffi_type_pointer);
                    set_value(&param_idx_);
                    increment();
                    set_arg_type(&ffi_type_pointer);
                    set_value(&param_size_);
                    increment();
                }
                void add_stream(void **stream) {
                    set_arg_type(&ffi_type_pointer);
                    streams_ = stream;
                    set_value(&streams_);
                    increment();
                }
                void add_dev_info() {
                    set_arg_type(&ffi_type_uint);
                    set_value(&dev_info_);
                    increment();
                }
                ffi_cif *cif() { return &cif_; }
                ffi_type **args() { return args_; }
                __iris_kernel_ptr fn_ptr() { return fn_ptr_; }
                Kernel *kernel() { return kernel_; }
                void set_dev_info(int32_t stream_index, int nstreams, int devno) { dev_info_ = HostInterfaceLoader::get_encoded_stream_device(stream_index, nstreams, devno); }
                void set_nstream(int32_t nstreams) { dev_info_ = HostInterfaceLoader::update_dev_info_nstreams(dev_info_, nstreams); }
                void set_devno(int32_t devno) { dev_info_ = HostInterfaceLoader::update_dev_info_devno(dev_info_, devno); }
                void set_stream_index(int32_t stream_index) { dev_info_ = HostInterfaceLoader::update_dev_info_stream_index(dev_info_, stream_index); }
            private:
                int args_capacity_;
                Kernel *kernel_;
                __iris_kernel_ptr fn_ptr_;
                KernelArg *iris_args_;
                ffi_type **args_;
                void **values_;
                void **values_ptr_;
                ffi_cif cif_;
                size_t *param_size_;
                size_t *param_idx_;
                FFICallType type_;
                void **streams_;
                uint32_t dev_info_;
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
                int launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd);
                int SetKernelPtr(void *obj, const char *kernel_name);
                int host_launch(void **stream, int stream_index, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws);
                int setarg(void *param_mem, int kindex, size_t size, void *value);
                int setmem(void *param_mem, int kindex, void *mem, size_t size);
            private:
                ffi_cif cif_;
        };
#endif
    }
}
