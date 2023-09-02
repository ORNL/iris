#include "HostInterface.h"
#include "Device.h"
#include "Platform.h"

#define INIT_SYM_FN(FN)  \
    sym_map_fn_.insert(make_pair(string(#FN), (void **)&FN)); \
    FN = NULL;
namespace iris {
    namespace rt {
        const char* HostInterfaceLoader::library() {
            char* path = NULL;
            Platform::GetPlatform()->EnvironmentGet(kernel_env_.c_str(), &path, NULL);
            return path;
        }
        BoilerPlateHostInterfaceLoader::BoilerPlateHostInterfaceLoader(string kernel_env):
            HostInterfaceLoader(kernel_env) {
                INIT_SYM_FN(iris_host_init);
                INIT_SYM_FN(iris_host_init_handles);
                INIT_SYM_FN(iris_host_finalize_handles);
                INIT_SYM_FN(iris_host_finalize);
                INIT_SYM_FN(iris_host_kernel);
                INIT_SYM_FN(iris_host_setarg);
                INIT_SYM_FN(iris_host_setmem);
                INIT_SYM_FN(iris_host_launch);
                INIT_SYM_FN(iris_host_kernel_with_obj);
                INIT_SYM_FN(iris_host_setarg_with_obj);
                INIT_SYM_FN(iris_host_setmem_with_obj);
                INIT_SYM_FN(iris_host_launch_with_obj);
                INIT_SYM_FN(iris_get_kernel_names);
                INIT_SYM_FN(iris_set_kernel_ptr_with_obj);
            }
        void BoilerPlateHostInterfaceLoader::LoadFunction(const char *func_name, const char *symbol) 
        {
            if (sym_map_fn_.find(func_name) != sym_map_fn_.end()) {
                *(sym_map_fn_[func_name]) = dlsym(handle(), symbol);
            }
        }
        void BoilerPlateHostInterfaceLoader::LoadFunctionStatic(const char *func_name, void *symbol) 
        {
            if (sym_map_fn_.find(func_name) != sym_map_fn_.end()) {
                *(sym_map_fn_[func_name]) = symbol;
            }
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
        void BoilerPlateHostInterfaceLoader::setmem(void *param_mem, int kindex, void *dev_ptr) 
        {
            if (iris_host_setmem_with_obj) 
                iris_host_setmem_with_obj(
                        param_mem, kindex, dev_ptr);
            else if (iris_host_setmem) 
                iris_host_setmem(kindex, dev_ptr);
        }
        void BoilerPlateHostInterfaceLoader::finalize() 
        {
            if (iris_host_finalize){
                iris_host_finalize();
            }
            if (iris_host_finalize_handles){
                iris_host_finalize_handles(dev());
            }
        }
        void BoilerPlateHostInterfaceLoader::init() 
        {
            if (iris_host_init != NULL) {
                iris_host_init();
            }
            if (iris_host_init_handles != NULL) {
                iris_host_init_handles(dev());
            }
        }
    }
}
