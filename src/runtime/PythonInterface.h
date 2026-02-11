#pragma once

#include "HostInterface.h"
#include "Debug.h"

namespace iris {
    namespace rt {
class KernelPython {
public:
    KernelPython()
        : kernel_(nullptr), iris_args_(nullptr), index_(0),
          python_kernel_type_(iris_python_native),  // or iris_python_native if you define it
          params_(nullptr),
          n_params_(0)
    {
    }

    void reset() {
        index_ = 0;

        // Reset kernel kind
        python_kernel_type_ = iris_python_native;   // adjust if you rename the enum/constant

        // Forget the parameter list
        params_   = nullptr;
        n_params_ = 0;

        // Zero the launch spec
    }
    void set_n_dims(int32_t dims) {
        assert(index_ < n_params_);
        params_[index_].n_dims = dims;
    }
    void set_arg_type(int32_t arg_type) {
        assert(index_ < n_params_);
        params_[index_].data_type = arg_type;
        //printf("Args:%p %d 0x%x\n", args_, index_, arg_type);
    }
    void set_value(void *value) { 
        assert(index_ < n_params_);
        params_[index_].data = value;
        //printf("         Values ptr:%p\n", values_+index_); 
    }
    void set_value_ptr(void *value) { 
        assert(index_ < n_params_);
        params_[index_].data = value;
        //printf("         Values mem:%p mem_ptr:%p\n", values_ptr_[index_], values_[index_]); 
    }
    void set_element_size(size_t size) {
        assert(index_ < n_params_);
        params_[index_].element_size = size;
    }
    void set_dim_off(size_t *sizes) { 
        assert(index_ < n_params_);
        params_[index_].off = sizes;
    }
    void set_dim_sizes(size_t *sizes) { 
        assert(index_ < n_params_);
        params_[index_].tile_sizes= sizes;
    }
    void set_dim_full_sizes(size_t *sizes) { 
        assert(index_ < n_params_);
        params_[index_].full_sizes = sizes;
    }
    void set_count(size_t n)
    {
        assert(index_ < n_params_);
        params_[index_].count = n;
    }
    void set_is_pointer(int32_t p) {
        assert(index_ < n_params_);
        params_[index_].is_pointer = p;
    }
    void set_params(IrisParam *params, int n_params) {
        params_   = params;
        n_params_ = n_params;
    }
    void set_mem_id(unsigned long mem_id) {
        assert(index_ < n_params_);
        params_[index_].mem_id = mem_id;
    }

    IrisParam       *params()       { return params_; }
    const IrisParam *params() const { return params_; }
    int              n_params() const { return n_params_; }


    int python_kernel_type() const { return python_kernel_type_; }
    void set_python_kernel_type(int kind) { python_kernel_type_ = kind; }
    void init(int nargs) {
        n_params_ = nargs;
        if (n_params_ > 0) {
            params_ = new IrisParam[n_params_];
        }
        index_  = 0;
    }
    int top() { return index_; }
    void increment() { index_++; }
    void set_kernel(Kernel *kernel) { kernel_ = kernel; }
    Kernel *kernel() { return kernel_; }
    void set_iris_args(KernelArg *iris_args) { iris_args_ = iris_args; }
    KernelArg *get_iris_arg(int kindex) { return iris_args_ + kindex; }
private:
    Kernel *kernel_;
    KernelArg *iris_args_;
    int       index_;
    int       python_kernel_type_;  // renamed from python_kernel_type_

    IrisParam *params_;
    int        n_params_;
};
class PythonHostInterfaceLoader : public HostInterfaceLoader {
    public:
        PythonHostInterfaceLoader(int model);
        ~PythonHostInterfaceLoader() { }
        const char* library() { return "libpython.so"; }
        void init(int dev);
        int LoadFunctions();
        int host_kernel(void *param_mem, const char *kname);
        void set_kernel_python(void *param_mem, KernelPython *python_data);
        KernelPython * get_kernel_python(void *param_mem);
        int launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd);
        int SetKernelPtr(void *obj, const char *kernel_name);
        int host_launch(void **stream, int stream_index, int nstreams, const char *kname, void *param_mem, int devno, int dim, size_t *off, size_t *bws) { _error("Not implemented"); return IRIS_ERROR; }
        int host_launch(unsigned long task_id, unsigned long cmd_id, void **stream, int stream_index, void *ctx, bool async, int nstreams, const char *kname, void *py_fn_ptr, void *param_mem, int devno, int dim, size_t *gws, size_t *off, size_t *bws, size_t shared_mem_size);
        int setarg(void *param_mem, int kindex, size_t size, void *value);
        int setmem(void *param_mem, BaseMem *mem, int kindex, void *mem_ptr, size_t size);
        int set_python_kernel_type(void *param_mem, int type);
        void launch_python_kernel(void *kname, IrisSpec *spec, IrisParam *params, int n)
        {
            if (iris_get_python_launch_func()!= NULL) {
                python_kernel_t kernel_python_wrapper = iris_get_python_launch_func();
                kernel_python_wrapper(kname, spec, params, n);
            }
        }
    private:
        int target_;
};
} // rt
} // iris
