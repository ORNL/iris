#include "PythonInterface.h"
#include "Device.h"
#include "Command.h"
#include "Kernel.h"
#include "Platform.h"
#include "signal.h"
#include "DataMem.h"


namespace iris {
    namespace rt {
PythonHostInterfaceLoader::PythonHostInterfaceLoader(int target) :
    HostInterfaceLoader() {
        target_ = target;
    }
int PythonHostInterfaceLoader::LoadFunctions() {
    return IRIS_SUCCESS;
}
KernelPython *PythonHostInterfaceLoader::get_kernel_python(void *param_mem) {
    ASSERT(sizeof(KernelPython) < KERNEL_ARGS_MEM_SIZE);
    KernelPython **python_ptr = (KernelPython **)(((uint8_t *)param_mem)+ sizeof(int));
    return *python_ptr;
}
void PythonHostInterfaceLoader::set_kernel_python(void *param_mem, KernelPython *python_data)
{
    ASSERT(sizeof(KernelPython) < KERNEL_ARGS_MEM_SIZE);
    KernelPython **python_ptr = (KernelPython **)(((uint8_t *)param_mem)+ sizeof(int));
    //printf("Python data: %p %p : %p\n", param_mem, python_ptr, python_data);
    *python_ptr = python_data;
}
int PythonHostInterfaceLoader::launch_init(int model, int devno, int stream_index, int nstreams, void **stream, void *param_mem, Command *cmd) 
{
    Kernel *kernel = cmd->kernel();
    const char *name = kernel->name();
    kernel->CreatePythonData(sizeof(KernelPython));
    KernelPython *python_data = (KernelPython *) kernel->GetPythonData();
    //printf("Python Data:%p size:%lu\n", python_data, sizeof(KernelPython));
    set_kernel_python(param_mem, python_data);
    python_data->reset();
    python_data->init(cmd->kernel_nargs());
    python_data->set_kernel(kernel);
    python_data->set_iris_args(cmd->kernel_args());
    //printf("Python param_size:%p\n", python_data->param_size());
    //python_data->set_kernel(kernel);
    return IRIS_SUCCESS;
}
void PythonHostInterfaceLoader::init(int dev) 
{
    //HostInterfaceLoader::init(dev);
    //printf("Initialized Python init\n");
    //printf("Loading python library\n");
    //py_value_t *jfn = (py_value_t *)iris_get_python_launch_func();
}
int PythonHostInterfaceLoader::host_kernel(void *param_mem, const char *kname) 
{
    return IRIS_SUCCESS;
}
int PythonHostInterfaceLoader::set_python_kernel_type(void *obj, int type)
{
    KernelPython *python_data = get_kernel_python(obj);
    python_data->set_python_kernel_type(type);
    return IRIS_SUCCESS;
}
int PythonHostInterfaceLoader::SetKernelPtr(void *obj, const char *kernel_name)
{
    KernelPython *python_data = get_kernel_python(obj);
    //python_data->set_kernel_name(kernel_name);
    return IRIS_SUCCESS;
}
int PythonHostInterfaceLoader::host_launch(unsigned long task_id, unsigned long cmd_id, void **stream, int stream_index, void *ctx, bool async, int nstreams, const char *kname, void *py_fn_ptr, void *param_mem, int devno, int dim, size_t *gws, size_t *grid, size_t *block, size_t shared_mem_size)
{
    KernelPython *python_data = get_kernel_python(param_mem);
    IrisSpec spec = { nstreams, devno, devno, async, task_id, cmd_id, stream, ctx, dim, gws, grid, block, shared_mem_size }; 
    SetKernelPtr(param_mem, kname);
    launch_python_kernel(py_fn_ptr, &spec, python_data->params(), python_data->n_params());
    return IRIS_SUCCESS;
}
int PythonHostInterfaceLoader::setarg(void *param_mem, int kindex, size_t size, void *value)
{
    KernelPython *python_data = get_kernel_python(param_mem);
    KernelArg *arg = python_data->get_iris_arg(kindex);
    //printf("setarg python_data:%p kindex:%d arg_index:%d size:%lu value:%p\n", python_data, kindex, python_data->top(), size, value);
    if (arg->data_type != iris_unknown) 
        python_data->set_arg_type(arg->data_type << 8);
    else if (size == 1) 
        python_data->set_arg_type(iris_uint8 << 8);
    else if (size == 2) 
        python_data->set_arg_type(iris_uint16 << 8);
    else if (size == 4) 
        python_data->set_arg_type(iris_uint32 << 8);
    else if (size == 8) 
        python_data->set_arg_type(iris_uint64 << 8);
    else {
        _error("Unknown argument size:%lu at index for kernel:%s", size, python_data->kernel()->name());
    }
    python_data->set_value(value);
    python_data->increment();
    return IRIS_SUCCESS;
}
int PythonHostInterfaceLoader::setmem(void *param_mem, BaseMem *mem, int kindex, void *mem_ptr, size_t size)
{
    KernelPython *python_data = get_kernel_python(param_mem);
    int element_type = mem->element_type();
    python_data->set_mem_id(mem->uid());
    MemHandlerType h_type = mem->GetMemHandlerType();
    int element_size = 0;
    int dim = 1;
    //int dev_tile_dim = 1;
    //int preserve_size_flag = 0;
    if (h_type == IRIS_DMEM_REGION || h_type == IRIS_DMEM) {
        DataMem *dmem = (DataMem *)mem;
        element_size = (int) dmem->elem_size();
        dim = dmem->dim();
        python_data->set_n_dims(dim);
        //dev_tile_dim = dmem->dev_tile_dim();
        //printf("Size DIM : %lu\n", dmem->dev_size()[0]);
        python_data->set_dim_off(dmem->dev_off());
        python_data->set_dim_sizes(dmem->dev_tile_size());
        python_data->set_dim_full_sizes(dmem->dev_size());
    }
    python_data->set_arg_type(element_type);
    python_data->set_is_pointer(iris_pointer);
    python_data->set_element_size(element_size);

    //printf("setmem python_data:%p kindex:%d arg_index:%d mem:%p dev:%p size:%ld element_size:%d\n", python_data, kindex, python_data->top(), mem, mem_ptr, size, element_size);
    python_data->set_value_ptr(mem_ptr);
    python_data->set_count(size/element_size);
    python_data->increment();
    //raise(SIGABRT);
    return IRIS_SUCCESS;
}
} // rt
} // iris
