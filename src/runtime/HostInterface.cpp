#include "HostInterface.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Platform.h"
#include "signal.h"
#include "DataMem.h"

#define INIT_SYM_FN(FN)  \
    sym_map_fn_.insert(make_pair(string(#FN), (void **)&FN)); \
    FN = NULL;
namespace iris {
    namespace rt {
        HostInterfaceLoader::HostInterfaceLoader(string kernel_env) : Loader(), kernel_env_(kernel_env) 
        { 
            INIT_SYM_FN(iris_host_init);
            INIT_SYM_FN(iris_host_init_handles);
            INIT_SYM_FN(iris_host_finalize_handles);
            INIT_SYM_FN(iris_host_finalize);
            INIT_SYM_FN(iris_host_launch);
            INIT_SYM_FN(iris_host_launch_with_obj);
            INIT_SYM_FN(iris_ffi_launch);
        }
        uint32_t HostInterfaceLoader::get_encoded_stream_device(int stream_index, int nstreams, int devno)
        {
            ASSERT(stream_index < 256 && "Stream index should be <256");
            ASSERT(nstreams < 256 && "Stream index should be <256");
            ASSERT(devno < 65536 && "Stream index should be <256");
            uint32_t dev_info = ((uint32_t)(nstreams << 24)) | (((uint32_t)stream_index) << 16) | devno;
            return dev_info;
        }
        uint32_t HostInterfaceLoader::update_dev_info_nstreams(int dev_info, int nstreams)
        {
            uint32_t stream_index = (dev_info >> 16) & 0xFF;
            //uint32_t nstreams = (dev_info >> 24);
            uint32_t devno = (dev_info & 0xFFFF);
            return get_encoded_stream_device(stream_index, nstreams, devno);
        }
        uint32_t HostInterfaceLoader::update_dev_info_stream_index(int dev_info, int stream_index)
        {
            //uint32_t stream_index = (dev_info >> 16) & 0xFF;
            uint32_t nstreams = (dev_info >> 24);
            uint32_t devno = (dev_info & 0xFFFF);
            return get_encoded_stream_device(stream_index, nstreams, devno);
        }
        uint32_t HostInterfaceLoader::update_dev_info_devno(int dev_info, int devno)
        {
            uint32_t stream_index = (dev_info >> 16) & 0xFF;
            uint32_t nstreams = (dev_info >> 24);
            //uint32_t devno = (dev_info & 0xFFFF);
            return get_encoded_stream_device(stream_index, nstreams, devno);
        }
        void HostInterfaceLoader::init(int dev) 
        {
            if (iris_host_init != NULL) {
                iris_host_init();
            }
            if (iris_host_init_handles != NULL) {
                iris_host_init_handles(dev);
            }
        }
        void HostInterfaceLoader::finalize(int dev) 
        {
            if (iris_host_finalize){
                iris_host_finalize();
            }
            if (iris_host_finalize_handles){
                iris_host_finalize_handles(dev);
            }
        }
        const char* HostInterfaceLoader::library() {
            char* path = NULL;
            Platform::GetPlatform()->EnvironmentGet(kernel_env_.c_str(), &path, NULL);
            return path;
        }
#ifdef DISABLE_DYNAMIC_LINKING
        bool HostInterfaceLoader::IsFunctionExists(const char *kernel_name) {
            int kernel_idx = -1;
            if (iris_host_kernel_with_obj != NULL && 
                    iris_host_kernel_with_obj(&kernel_idx, kernel_name) == IRIS_SUCCESS) {
                return true;
            }
            if (iris_host_kernel != NULL && 
                    iris_host_kernel(kernel_name) == IRIS_SUCCESS)
                return true;
            return false;
        }
        void *HostInterfaceLoader::GetFunctionPtr(const char *kernel_name) {
            void *kptr = this->iris_host_get_kernel_ptr(kernel_name);
            return kptr;
        }
#endif
        void HostInterfaceLoader::LoadFunction(const char *func_name, const char *symbol) 
        {
            if (sym_map_fn_.find(string(func_name)) != sym_map_fn_.end()) {
                *(sym_map_fn_[string(func_name)]) = dlsym(handle(), symbol);
            }
        }
        void HostInterfaceLoader::LoadFunctionStatic(const char *func_name, void *symbol) 
        {
            if (sym_map_fn_.find(string(func_name)) != sym_map_fn_.end()) {
                *(sym_map_fn_[string(func_name)]) = symbol;
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
        int BoilerPlateHostInterfaceLoader::launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd) 
        {
            const char *kname = cmd->kernel()->name();
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
        int BoilerPlateHostInterfaceLoader::host_kernel(void *param_mem, const char *kname) 
        {
            if (iris_host_kernel_with_obj || iris_host_kernel) 
                if (IsFunctionExists(kname)) return IRIS_SUCCESS;
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
                kptr = (__iris_kernel_ptr) GetFunctionPtr(kernel_name);
                iris_set_kernel_ptr_with_obj(obj, kptr);
                if (kptr != NULL) return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        int BoilerPlateHostInterfaceLoader::host_launch(void **stream, int stream_index, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws) 
        {
            if(iris_host_launch_with_obj) {
                uint32_t dev_info = get_encoded_stream_device(stream_index, nstreams, devno);
                SetKernelPtr(param_mem, kname);
                int status = iris_host_launch_with_obj(
                        stream, 
                        param_mem, dev_info,  
                        dim, off, bws);
                return IRIS_SUCCESS;
            }
            else if(iris_host_launch) {
                int status = iris_host_launch(
                        dim, off, bws);
                return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        int BoilerPlateHostInterfaceLoader::setarg(void *param_mem, int kindex, size_t size, void *value) 
        {
            if (iris_host_setarg_with_obj)
                return iris_host_setarg_with_obj(
                        param_mem, kindex, size, value);
            else if (iris_host_setarg)
                return iris_host_setarg(kindex, size, value);
            return IRIS_ERROR;
        }
        int BoilerPlateHostInterfaceLoader::setmem(void *param_mem, int kindex, void *mem, size_t size) 
        {
            //printf("iris_host_setmem:%p kindex:%d\n", iris_host_setmem, kindex);
            if (iris_host_setmem_with_obj) 
                return iris_host_setmem_with_obj(
                        param_mem, kindex, mem, size);
            else if (iris_host_setmem) 
                return iris_host_setmem(kindex, mem);
            return IRIS_ERROR;
        }
        JuliaHostInterfaceLoader::JuliaHostInterfaceLoader(int target) :
            HostInterfaceLoader("KERNEL_JULIA") {
                target_ = target;
                jl_init = NULL;
                jl_atexit_hook = NULL;
                jl_is_initialized = NULL;
                jl_gc_add_finalizer = NULL;
                jl_unbox_voidpointer = NULL;
            }
        int JuliaHostInterfaceLoader::LoadFunctions() {
            HostInterfaceLoader::LoadFunctions();
            //printf("Loading Julia functions\n");
            LOADFUNC(jl_init);
            LOADFUNC(jl_atexit_hook);
            LOADFUNC(jl_is_initialized);
            LOADFUNC(jl_gc_add_finalizer);
            LOADFUNC(jl_unbox_voidpointer);
            return IRIS_SUCCESS;
        }
        KernelJulia *JuliaHostInterfaceLoader::get_kernel_julia(void *param_mem) {
            ASSERT(sizeof(KernelJulia) < KERNEL_ARGS_MEM_SIZE);
            KernelJulia **julia_ptr = (KernelJulia **)(((uint8_t *)param_mem)+ sizeof(int));
            return *julia_ptr;
        }
        void JuliaHostInterfaceLoader::set_kernel_julia(void *param_mem, KernelJulia *julia_data)
        {
            ASSERT(sizeof(KernelJulia) < KERNEL_ARGS_MEM_SIZE);
            KernelJulia **julia_ptr = (KernelJulia **)(((uint8_t *)param_mem)+ sizeof(int));
            //printf("Julia data: %p %p : %p\n", param_mem, julia_ptr, julia_data);
            *julia_ptr = julia_data;
        }
        int JuliaHostInterfaceLoader::launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd) 
        {
            Kernel *kernel = cmd->kernel();
            const char *name = kernel->name();
            kernel->CreateJuliadata(sizeof(KernelJulia));
            KernelJulia *julia_data = (KernelJulia *) kernel->GetJuliadata();
            //printf("Julia Data:%p size:%lu\n", julia_data, sizeof(KernelJulia));
            set_kernel_julia(param_mem, julia_data);
            julia_data->reset();
            julia_data->init(cmd->kernel_nargs());
            //printf("Julia param_size:%p\n", julia_data->param_size());
            julia_data->set_kernel(kernel);
            julia_data->set_iris_args(cmd->kernel_args());
            julia_data->add_stream(stream); 
            julia_data->set_dev_info(stream_index, nstreams, devno);
            julia_data->add_dev_info(); 
            return IRIS_SUCCESS;
        }
        void JuliaHostInterfaceLoader::init(int dev) 
        {
            HostInterfaceLoader::init(dev);
            //printf("Initialized Julia init\n");
            //printf("Loading julia library\n");
            if (Load() != IRIS_SUCCESS) {
              _trace("%s", "skipping Julia wrapper calls");
            }
            //jl_value_t *jfn = (jl_value_t *)iris_get_julia_launch_func();
           // jl_gc_add_finalizer(jfn, NULL);
            (*jl_init)();
        }
        int JuliaHostInterfaceLoader::host_kernel(void *param_mem, const char *kname) 
        {
            return IRIS_SUCCESS;
        }
        int JuliaHostInterfaceLoader::set_julia_kernel_type(void *obj, int type)
        {
            KernelJulia *julia_data = get_kernel_julia(obj);
            julia_data->set_julia_kernel_type(type);
            return IRIS_SUCCESS;
        }
        int JuliaHostInterfaceLoader::SetKernelPtr(void *obj, const char *kernel_name)
        {
            KernelJulia *julia_data = get_kernel_julia(obj);
            //julia_data->set_kernel_name(kernel_name);
            return IRIS_SUCCESS;
        }
        int JuliaHostInterfaceLoader::host_launch(unsigned long task_id, void **stream, int stream_index, void *ctx, bool async, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *grid, size_t *block)
        {
            KernelJulia *julia_data = get_kernel_julia(param_mem);
            SetKernelPtr(param_mem, kname);
            launch_julia_kernel(task_id, julia_data->julia_kernel_type(), target_, devno, ctx, async, stream_index, stream, nstreams, julia_data->args(), julia_data->values(), julia_data->param_size(), julia_data->param_dim_size(), julia_data->top(), grid, block, dim, kname);
            return IRIS_SUCCESS;
        }
        int JuliaHostInterfaceLoader::setarg(void *param_mem, int kindex, size_t size, void *value)
        {
            KernelJulia *julia_data = get_kernel_julia(param_mem);
            KernelArg *arg = julia_data->get_iris_arg(kindex);
            //printf("setarg julia_data:%p kindex:%d arg_index:%d size:%lu value:%p\n", julia_data, kindex, julia_data->top(), size, value);
            if (arg->data_type != iris_unknown) 
                julia_data->set_arg_type(arg->data_type);
            else if (size == 1) 
                julia_data->set_arg_type(iris_uint8);
            else if (size == 2) 
                julia_data->set_arg_type(iris_uint16);
            else if (size == 4) 
                julia_data->set_arg_type(iris_uint32);
            else if (size == 8) 
                julia_data->set_arg_type(iris_uint64);
            else {
                _error("Unknown argument size:%lu at index for kernel:%s", size, julia_data->kernel()->name());
            }
            julia_data->set_value(value);
            julia_data->increment();
            return IRIS_SUCCESS;
        }
        int JuliaHostInterfaceLoader::setmem(void *param_mem, BaseMem *mem, int kindex, void *mem_ptr, size_t size)
        {
            KernelJulia *julia_data = get_kernel_julia(param_mem);
            int element_type = mem->element_type();
            MemHandlerType h_type = mem->GetMemHandlerType();
            int element_size = 0;
            int dim = 1;
            if (h_type == IRIS_DMEM_REGION || h_type == IRIS_DMEM) {
                DataMem *dmem = (DataMem *)mem;
                element_size = (int) dmem->elem_size();
                dim = dmem->dim();
                //printf("Size DIM : %lu\n", dmem->dev_size()[0]);
                julia_data->set_dim_sizes(dmem->dev_size(), dmem->dim());
            }
            //printf("setmem julia_data:%p kindex:%d arg_index:%d mem:%p dev:%p size:%ld element_size:%d\n", julia_data, kindex, julia_data->top(), mem, mem_ptr, size, element_size);
            julia_data->set_arg_type(iris_pointer | element_type | (element_size << 8) | dim);
            julia_data->set_value_ptr(mem_ptr);
            julia_data->set_size(size/element_size);
            julia_data->increment();
            //raise(SIGABRT);
            return IRIS_SUCCESS;
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
        int FFIHostInterfaceLoader::launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd) 
        {
            Kernel *kernel = cmd->kernel();
            const char *name = kernel->name();
            if (! IsFunctionExists(name)) return IRIS_ERROR;
            kernel->CreateFFIdata(sizeof(KernelFFI));
            KernelFFI *ffi_data = (KernelFFI *) kernel->GetFFIdata();
            //printf("FFI Data:%p size:%lu\n", ffi_data, sizeof(KernelFFI));
            set_kernel_ffi(param_mem, ffi_data);
            ffi_data->init(cmd->kernel_nargs());
            ffi_data->set_kernel(kernel);
            ffi_data->set_iris_args(cmd->kernel_args());
            if (iris_host_launch_with_obj) {
                ffi_data->set_host_launch_type(HOST_LAUNCH_WITH_STREAM_DEV);
                if (model != iris_openmp) {
                    ffi_data->add_stream(stream); 
                    ffi_data->set_dev_info(stream_index, nstreams, devno);
                    //ffi_data->set_stream_index(stream_index);
                    //ffi_data->set_devno(devno);
                    //ffi_data->set_nstream(nstreams);
                    ffi_data->add_dev_info(); 
                }
            }
            else if (iris_host_launch) {
                ffi_data->set_host_launch_type(HOST_LAUNCH_BARE);
                //ffi_data->add_stream(stream); 
            }
            else if (iris_ffi_launch)
                ffi_data->set_host_launch_type(HOST_LAUNCH_WITH_FFI_OBJ);
            else 
                ffi_data->set_host_launch_type(HOST_LAUNCH_FFI_BARE);
            return IRIS_SUCCESS;
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
                kptr = (__iris_kernel_ptr) GetFunctionPtr(kernel_name);
                KernelFFI *ffi_data = get_kernel_ffi(obj);
                ffi_data->set_fn_ptr(kptr);
                if (kptr != NULL) return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        int FFIHostInterfaceLoader::host_launch(void **stream, int stream_index, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws)
        {
            if (IsFunctionExists(kname)) {
                KernelFFI *ffi_data = get_kernel_ffi(param_mem);
                SetKernelPtr(param_mem, kname);
                //ffi_data->add_epilog(off[0], bws[0]);
                ffi_data->add_epilog(off, bws);
                ffi_arg rc;
                ffi_prep_cif(ffi_data->cif(), FFI_DEFAULT_ABI, ffi_data->top(), &ffi_type_sint, ffi_data->args());
                ffi_call(ffi_data->cif(), ffi_data->fn_ptr(), &rc, ffi_data->values());
                return IRIS_SUCCESS;
            }
            return IRIS_ERROR;
        }
        int FFIHostInterfaceLoader::setarg(void *param_mem, int kindex, size_t size, void *value)
        {
            KernelFFI *ffi_data = get_kernel_ffi(param_mem);
            KernelArg *arg = ffi_data->get_iris_arg(kindex);
            //printf("setarg ffi_data:%p kindex:%d arg_index:%d size:%lu value:%p\n", ffi_data, kindex, ffi_data->top(), size, value);
            if (size == 1) 
                ffi_data->set_arg_type(&ffi_type_uint8);
            else if (size == 2) 
                ffi_data->set_arg_type(&ffi_type_uint16);
            else if (size == 4 && arg->data_type == iris_float)  // Special case to handle for double
                ffi_data->set_arg_type(&ffi_type_float);
            else if (size == 4) 
                ffi_data->set_arg_type(&ffi_type_uint32);
            else if (size == 8 && arg->data_type == iris_double)  // Special case to handle for double
                ffi_data->set_arg_type(&ffi_type_double);
            else if (size == 8) 
                ffi_data->set_arg_type(&ffi_type_uint64);
            else {
                _error("Unknown argument size:%lu at index for kernel:%s", size, ffi_data->kernel()->name());
            }
            ffi_data->set_value(value);
            ffi_data->increment();
            return IRIS_SUCCESS;
        }
        int FFIHostInterfaceLoader::setmem(void *param_mem, int kindex, void *mem, size_t size)
        {
            KernelFFI *ffi_data = get_kernel_ffi(param_mem);
            //printf("setmem ffi_data:%p kindex:%d arg_index:%d mem:%p\n", ffi_data, kindex, ffi_data->top(), mem);
            ffi_data->set_arg_type(&ffi_type_pointer);
            ffi_data->set_value_ptr(mem);
            ffi_data->increment();
            //raise(SIGABRT);
            return IRIS_SUCCESS;
        }
#endif
    }
}
