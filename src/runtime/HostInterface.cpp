#include "HostInterface.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Platform.h"
#include "signal.h"

#define INIT_SYM_FN(FN)  \
    sym_map_fn_.insert(make_pair(string(#FN), (void **)&FN)); \
    FN = NULL;
namespace iris {
    namespace rt {
        HostInterfaceLoader::HostInterfaceLoader(string kernel_env) : Loader(), kernel_env_(kernel_env), dev_(0) 
        { 
            INIT_SYM_FN(iris_host_init);
            INIT_SYM_FN(iris_host_init_handles);
            INIT_SYM_FN(iris_host_finalize_handles);
            INIT_SYM_FN(iris_host_finalize);
            INIT_SYM_FN(iris_host_launch);
            INIT_SYM_FN(iris_ffi_launch);
            INIT_SYM_FN(iris_host_launch_with_obj);
        }
        void HostInterfaceLoader::init() 
        {
            if (iris_host_init != NULL) {
                iris_host_init();
            }
            if (iris_host_init_handles != NULL) {
                iris_host_init_handles(dev());
            }
        }
        void HostInterfaceLoader::finalize() 
        {
            if (iris_host_finalize){
                iris_host_finalize();
            }
            if (iris_host_finalize_handles){
                iris_host_finalize_handles(dev());
            }
        }
        const char* HostInterfaceLoader::library() {
            char* path = NULL;
            Platform::GetPlatform()->EnvironmentGet(kernel_env_.c_str(), &path, NULL);
            return path;
        }
        void HostInterfaceLoader::LoadFunction(const char *func_name, const char *symbol) 
        {
            if (sym_map_fn_.find(func_name) != sym_map_fn_.end()) {
                *(sym_map_fn_[func_name]) = dlsym(handle(), symbol);
            }
        }
        void HostInterfaceLoader::LoadFunctionStatic(const char *func_name, void *symbol) 
        {
            if (sym_map_fn_.find(func_name) != sym_map_fn_.end()) {
                *(sym_map_fn_[func_name]) = symbol;
            }
        }
        BoilerPlateHostInterfaceLoader::BoilerPlateHostInterfaceLoader(string kernel_env):
            HostInterfaceLoader(kernel_env) {
                INIT_SYM_FN(iris_host_kernel);
                INIT_SYM_FN(iris_host_setarg);
                INIT_SYM_FN(iris_host_setmem);
                INIT_SYM_FN(iris_host_kernel_with_obj);
                INIT_SYM_FN(iris_host_setarg_with_obj);
                INIT_SYM_FN(iris_host_setmem_with_obj);
                INIT_SYM_FN(iris_get_kernel_names);
                INIT_SYM_FN(iris_set_kernel_ptr_with_obj);
            }
        int BoilerPlateHostInterfaceLoader::host_kernel(void *param_mem, const char *kname) 
        {
            if (iris_host_kernel_with_obj) {
                int status = iris_host_kernel_with_obj(
                        param_mem, kname);
                if (status == IRIS_SUCCESS && 
                        IsFunctionExists(kname)) {
                    return IRIS_SUCCESS;
                }
            }
            else if (iris_host_kernel) {
                int status = iris_host_kernel(kname);
                if (status == IRIS_SUCCESS && 
                        IsFunctionExists(kname)) {
                    return IRIS_SUCCESS;
                }
            }
            return IRIS_ERROR;
        }
        int BoilerPlateHostInterfaceLoader::LoadFunctions()
        {
            HostInterfaceLoader::LoadFunctions();
            REGISTER_HOST_WRAPPER(iris_get_kernel_names,      iris_get_kernel_names);
            REGISTER_HOST_WRAPPER(iris_set_kernel_ptr_with_obj, iris_set_kernel_ptr_with_obj);
            return IRIS_SUCCESS;
        }
        int BoilerPlateHostInterfaceLoader::SetKernelPtr(void *obj, const char *kernel_name)
        {
            if (iris_set_kernel_ptr_with_obj) {
                __iris_kernel_ptr kptr;
                kptr = (__iris_kernel_ptr) dlsym(handle(), kernel_name);
                iris_set_kernel_ptr_with_obj(obj, kptr);
                if (kptr != NULL) return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        int BoilerPlateHostInterfaceLoader::host_launch(void *stream, const char *kname, void *param_mem, int dim, size_t *off, size_t *gws) 
        {
            if(iris_host_launch_with_obj) {
                SetKernelPtr(param_mem, kname);
                int status = iris_host_launch_with_obj(
#ifndef IRIS_SYNC_EXECUTION
                        stream, 
#endif
                        param_mem, dev(),  
                        dim, off[0], gws[0]);
                return IRIS_SUCCESS;
            }
            else if(iris_host_launch) {
                int status = iris_host_launch(
#ifndef IRIS_SYNC_EXECUTION
                        stream, 
#endif
                        dim, off[0], gws[0]);
                return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        void BoilerPlateHostInterfaceLoader::setarg(void *param_mem, int kindex, size_t size, void *value) 
        {
            if (iris_host_setarg_with_obj)
                iris_host_setarg_with_obj(
                        param_mem, kindex, size, value);
            else if (iris_host_setarg)
                iris_host_setarg(kindex, size, value);
        }
        void BoilerPlateHostInterfaceLoader::setmem(void *param_mem, int kindex, void *mem, void **mem_ptr) 
        {
            if (iris_host_setmem_with_obj) 
                iris_host_setmem_with_obj(
                        param_mem, kindex, mem);
            else if (iris_host_setmem) 
                iris_host_setmem(kindex, mem);
        }
#ifdef ENABLE_FFI
        FFIHostInterfaceLoader::FFIHostInterfaceLoader(string kernel_env) :
            HostInterfaceLoader(kernel_env) {
                
            }
        int FFIHostInterfaceLoader::LoadFunctions() {
            HostInterfaceLoader::LoadFunctions();
            return IRIS_SUCCESS;
        }
        KernelFFI *FFIHostInterfaceLoader::get_kernel_ffi(void *param_mem) {
            KernelFFI **ffi_ptr = (KernelFFI **)(((uint8_t *)param_mem)+ sizeof(int));
            return *ffi_ptr;
        }
        void FFIHostInterfaceLoader::set_kernel_ffi(void *param_mem, KernelFFI *ffi_data)
        {
            KernelFFI **ffi_ptr = (KernelFFI **)(((uint8_t *)param_mem)+ sizeof(int));
            *ffi_ptr = ffi_data;
        }
        void FFIHostInterfaceLoader::launch_init(void *stream, void *param_mem, Command *cmd) 
        {
            Kernel *kernel = cmd->kernel();
            const char *name = kernel->name();
            if (! IsFunctionExists(name)) return;
            kernel->CreateFFIdata(sizeof(KernelFFI));
            KernelFFI *ffi_data = (KernelFFI *) kernel->GetFFIdata();
            printf("FFI Data:%p size:%lu\n", ffi_data, sizeof(KernelFFI));
            set_kernel_ffi(param_mem, ffi_data);
            ffi_data->init(cmd->kernel_nargs());
            ffi_data->set_kernel(kernel);
            ffi_data->set_iris_args(cmd->kernel_args());
            if (iris_host_launch_with_obj) {
                ffi_data->set_host_launch_type(HOST_LAUNCH_WITH_STREAM_DEV);
#ifndef IRIS_SYNC_EXECUTION
                ffi_data->add_stream(stream); 
#endif
                ffi_data->add_device(dev_ptr()); 
            }
            else if (iris_host_launch) {
                ffi_data->set_host_launch_type(HOST_LAUNCH_BARE);
#ifndef IRIS_SYNC_EXECUTION
                ffi_data->add_stream(stream); 
#endif
            }
            else if (iris_ffi_launch)
                ffi_data->set_host_launch_type(HOST_LAUNCH_WITH_FFI_OBJ);
            else 
                ffi_data->set_host_launch_type(HOST_LAUNCH_FFI_BARE);
        }
        int FFIHostInterfaceLoader::host_kernel(void *param_mem, const char *kname) 
        {
            if (IsFunctionExists(kname)) 
                return IRIS_SUCCESS;
            return IRIS_ERROR;
        }
        int FFIHostInterfaceLoader::SetKernelPtr(void *obj, const char *kernel_name)
        {
            if (IsFunctionExists(kernel_name)) {
                __iris_kernel_ptr kptr;
                kptr = (__iris_kernel_ptr) dlsym(handle(), kernel_name);
                KernelFFI *ffi_data = get_kernel_ffi(obj);
                ffi_data->set_fn_ptr(kptr);
                if (kptr != NULL) return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        int FFIHostInterfaceLoader::host_launch(void *stream, const char *kname, void *param_mem, int dim, size_t *off, size_t *gws)
        {
            if (IsFunctionExists(kname)) {
                KernelFFI *ffi_data = get_kernel_ffi(param_mem);
                SetKernelPtr(param_mem, kname);
                ffi_data->add_epilog();
                ffi_arg rc;
                ffi_prep_cif(ffi_data->cif(), FFI_DEFAULT_ABI, ffi_data->top(), &ffi_type_sint, ffi_data->args());
                ffi_call(ffi_data->cif(), ffi_data->fn_ptr(), &rc, ffi_data->values());
            }
            return IRIS_ERROR;
        }
        void FFIHostInterfaceLoader::setarg(void *param_mem, int kindex, size_t size, void *value)
        {
            KernelFFI *ffi_data = get_kernel_ffi(param_mem);
            KernelArg *arg = ffi_data->get_iris_arg(kindex);
            printf("setarg ffi_data:%p kindex:%d arg_index:%d size:%lu value:%p\n", ffi_data, kindex, ffi_data->top(), size, value);
            if (size == 1) 
                ffi_data->set_arg_type(&ffi_type_uint8);
            else if (size == 2) 
                ffi_data->set_arg_type(&ffi_type_uint16);
            else if (size == 4 && arg->data_type == iris_float) 
                ffi_data->set_arg_type(&ffi_type_float);
            else if (size == 4) 
                ffi_data->set_arg_type(&ffi_type_uint32);
            else if (size == 8 && arg->data_type == iris_double) 
                ffi_data->set_arg_type(&ffi_type_double);
            else if (size == 8) 
                ffi_data->set_arg_type(&ffi_type_uint64);
            else {
                _error("Unknown argument size:%lu at index for kernel:%s", size, ffi_data->kernel()->name());
            }
            ffi_data->set_value(value);
            ffi_data->increment();
        }
        void FFIHostInterfaceLoader::setmem(void *param_mem, int kindex, void *mem, void **mem_ptr)
        {
            KernelFFI *ffi_data = get_kernel_ffi(param_mem);
            printf("setmem ffi_data:%p kindex:%d arg_index:%d mem:%p mem_ptr:%p\n", ffi_data, kindex, ffi_data->top(), mem, mem_ptr);
            ffi_data->set_arg_type(&ffi_type_pointer);
            ffi_data->set_value(mem_ptr);
            ffi_data->increment();
            //raise(SIGABRT);
        }
#endif
    }
}
