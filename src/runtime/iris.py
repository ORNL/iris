import ctypes
from ctypes import *
import sys
import numpy as np
import pdb
import os
import struct 

class CommData3D(Structure):
    _fields_ = [
        ("from_id", c_uint),
        ("to_id", c_uint), 
        ("mem_id", c_uint),
        ("size", c_size_t)
    ]
    def __repr__(self):
        return f'CommData3D(from_id={self.from_id}, to_id={self.to_id}, mem_id={self.mem_id}, size={self.size})'
    def __str__(self):
        return f"from_id={self.from_id}, to_id={self.to_id}, mem_id={self.mem_id}, size={self.size}"

class TaskProfile(Structure):
    _fields_ = [
        ("task", c_uint),
        ("device", c_uint),
        ("start", c_double),
        ("end", c_double)
    ]
    def __str__(self):
        return f"task={self.task}, device={self.device}, start={self.start}, end={self.end}"

    def __repr__(self) -> str:
        return f"TaskProfile(task={self.task}, device={self.device}, start={self.start}, end={self.end})"


class DataObjectProfile(Structure):
    _fields_ = [
        ("task", c_uint),
        ("data_object", c_uint),
        ("datatransfer_type", c_uint),
        ("from_device", c_uint),
        ("device", c_uint),
        ("start", c_double),
        ("end", c_double)
    ]
    def __str__(self):
        return f"task={self.task}, data_object={self.data_object}, datatransfer_type={self.datatransfer_type}, from_device={self.from_device}, device={self.device}, start={self.start}, end={self.end}"

    def __repr__(self) -> str:
        return f"DataObjectProfile(task={self.task}, data_object={self.data_object}, datatransfer_type={self.datatransfer_type}, from_device={self.from_device}, device={self.device}, start={self.start}, end={self.end})"



class library(CDLL):
    def __init__(self, library_name, read_symbols=False):
        super().__init__(library_name, mode=RTLD_GLOBAL)

    def convert_c_pointer_to_numpy(self, cptr, shape, ct=ctypes.c_double):
        U = np.ctypeslib.as_array(ctypes.cast(cptr, ctypes.POINTER(ct)), shape=shape)
        return U

    def call_ret_ptr(self, fn_name, *args):
        conv_args = [] 
        for each_arg in args:
            conv_args.append(convert_obj_ctype(each_arg)[0])
        if type(fn_name) == str:
            fn_name = self.__getitem__(fn_name)
        fn_name.restype = POINTER(c_void_p)
        r = fn_name(*conv_args)
        return r 

    def call_ret(self, fn_name, restype, *args):
        conv_args = [] 
        for each_arg in args:
            conv_args.append(convert_obj_ctype(each_arg)[0])
        if type(fn_name) == str:
            fn_name = self.__getitem__(fn_name)
        fn_name.restype = restype
        return fn_name(*conv_args)

    def call(self, fn_name, *args):
        conv_args = [] 
        for each_arg in args:
            conv_args.append(convert_obj_ctype(each_arg)[0])
        if type(fn_name) == str:
            fn_name = self.__getitem__(fn_name)
        fn_name(*conv_args)

class IRIS(library):
    def __init__(self):
        super(IRIS, self).__init__("libiris.so")
    def get_numpy_size(self, size):
        if type(size) != np.ndarray:
            size = np.array(size)
        return size
    def alloc_random_size_t(self, size, init=np.int64(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_size_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_size_t)
        return np_data, c_ptr
        
    def alloc_random_float(self, size, init=np.float32(0.0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_float, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_float)
        return np_data, c_ptr
        
    def alloc_random_double(self, size, init=np.float64(0.0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_double, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_double)
        return np_data, c_ptr
        
    def alloc_random_int64(self, size, init=np.int64(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_int64_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_int64)
        return np_data, c_ptr
        
    def alloc_random_int32(self, size, init=np.int32(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_int32_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_int32)
        return np_data, c_ptr

    def alloc_random_int16(self, size, init=np.int16(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_int16_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_short)
        return np_data, c_ptr

    def alloc_random_int8(self, size, init=np.int8(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_random_array_int8_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_byte)
        return np_data, c_ptr
        
    def alloc_size_t(self, size, init=np.int64(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_size_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_size_t)
        return np_data, c_ptr
        
    def alloc_float(self, size, init=np.float32(0.0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_float, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_float)
        return np_data, c_ptr
        
    def alloc_double(self, size, init=np.float64(0.0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_double, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_double)
        return np_data, c_ptr
        
    def alloc_int64(self, size, init=np.int64(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_int64_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_int64)
        return np_data, c_ptr
        
    def alloc_int32(self, size, init=np.int32(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_int32_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_int32)
        return np_data, c_ptr
        
    def alloc_int16(self, size, init=np.int16(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_int16_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_short)
        return np_data, c_ptr

    def alloc_int8(self, size, init=np.int8(0)):
        size = self.get_numpy_size(size)
        total_size = size.prod()
        c_ptr = dll.call_ret_ptr(dll.iris_allocate_array_int8_t, np.int32(total_size), init)
        np_data = dll.convert_c_pointer_to_numpy(c_ptr, size, ctypes.c_byte)
        return np_data, c_ptr

    def free(self, c_ptr):
        dll.call(dll.iris_free_array, c_ptr)
        
dll = IRIS()

def call(fn_name, *args):
    dll.call(fn_name, *args)

def call_ret_ptr(fn_name, *args):
    return dll.call_ret_ptr(fn_name, *args)

iris_ok         =   0
iris_err        =   -1

iris_default    =       (1 << 5)
iris_cpu        =       (1 << 6)
iris_nvidia     =       (1 << 7)
iris_amd        =       (1 << 8)
iris_gpu_intel  =       (1 << 9)
iris_gpu        =       (iris_nvidia | iris_amd | iris_gpu_intel)
iris_phi        =       (1 << 10)
iris_fpga       =       (1 << 11)
iris_hexagon    =       (1 << 12)
iris_dsp        =       (iris_hexagon)
iris_roundrobin =       (1 << 18)
iris_depend     =       (1 << 19)
iris_data       =       (1 << 20)
iris_profile    =       (1 << 21)
iris_random     =       (1 << 22)
iris_pending    =       (1 << 23)
iris_any        =       (1 << 24)
iris_all        =       (1 << 25)
iris_custom     =       (1 << 26)


iris_r          =   -1
iris_w          =   -2
iris_rw         =   -3
iris_flush      =   -4

iris_int        =   (1 << 0)
iris_long       =   (1 << 1)
iris_float      =   (1 << 2)
iris_double     =   (1 << 3)

iris_normal     =   (1 << 10)
iris_reduction  =   (1 << 11)
iris_sum        =   ((1 << 12) | iris_reduction)
iris_max        =   ((1 << 13) | iris_reduction)
iris_min        =   ((1 << 14) | iris_reduction)

iris_platform   =   0x1001
iris_vendor     =   0x1002
iris_name       =   0x1003
iris_type       =   0x1004

iris_dt_h2d     =        1
iris_dt_h2o     =        2
iris_dt_d2o     =        3
iris_dt_d2h     =        4
iris_dt_d2d     =        5
iris_dt_d2h_h2d =        6
iris_dt_error   =        0

IRIS_MEM = 0x1
IRIS_DMEM = 0x2
IRIS_DMEM_REGION = 0x4

class iris_kernel(Structure):
    _fields_ = [("class_obj", c_void_p)]

class iris_mem(Structure):
    _fields_ = [("class_obj", c_void_p)]

class iris_task(Structure):
    _fields_ = [("class_obj", c_void_p)]

class iris_graph(Structure):
    _fields_ = [("class_obj", c_void_p)]

def init(sync = 1):
    return dll.iris_init(0, None, c_int(sync))

def set_enable_profiler(flag=True):
    dll.iris_set_enable_profiler(int(flag))

def register_pin_memory(cptr, size):
    dll.iris_register_pin_memory(convert_obj_ctype(cptr)[0], convert_obj_ctype(size_t(size))[0])

def finalize():
    return dll.iris_finalize()

def synchronize():
    return dll.iris_synchronize()

def platform_count():
    i = c_int()
    dll.iris_platform_count(byref(i))
    return i.value

def platform_info(platform, param):
    s = (c_char * 64)()
    dll.iris_platform_info(c_int(platform), c_int(param), s, None)
    return s.value

def all_platform_info():
    platform_names = []
    for i in range(platform_count()):
        plt = platform_info(i, iris_name)
        platform_names.append(plt)
    return platform_names

def device_count():
    i = c_int()
    dll.iris_device_count(byref(i))
    return i.value

def device_info(device, param):
    s = (c_char * 64)()
    dll.iris_device_info(c_int(device), c_int(param), s, None)
    return s.value

def calibrate_communication_cost_matrix(data_size, iterations=1, pin_memory=True, units='B'):
    print("Calibrate communication cost matrix from IRIS platform")
    ndevs = device_count()+1
    data_np = np.zeros(ndevs*ndevs).astype(np.double)
    data_np_cp = convert_obj_ctype(data_np)[0]
    iterations = np.int32(iterations)
    dll.iris_calibrate_communication_cost(data_np_cp, 
            convert_obj_ctype(size_t(data_size))[0], 
            convert_obj_ctype(iterations)[0], 
            convert_obj_ctype(np.int32(pin_memory))[0])
    np.seterr(divide='ignore')
    out_data_np = data_size / data_np.reshape((ndevs, ndevs))
    if units == 'KB':
        out_data_np = out_data_np / 1000
    elif units == 'MB':
        out_data_np = out_data_np / 1000000
    elif units == 'GB':
        out_data_np = out_data_np / 1000000000
    elif units == 'TB':
        out_data_np = out_data_np / 1000000000000
    elif units == 'PB':
        out_data_np = out_data_np / 1000000000000000
    return out_data_np

def all_device_info():
    dev_names = []
    for i in range(device_count()):
        dev = device_info(i, iris_name)
        dev_names.append(dev)
    return dev_names

def mem_create(size):
    m = iris_mem()
    dll.iris_mem_create(c_size_t(size), byref(m))
    return m

def dmem_create(*args):
    if len(args) == 1:
        host = args[0][0]
        m = iris_mem()
        size = host.nbytes
        c_host = host.ctypes.data_as(c_void_p)
        dll.iris_data_mem_create(byref(m), c_host, c_size_t(size))
        return m
    elif len(args) == 2:
        (root_mem, host) = args
        m = iris_mem()
        dll.iris_data_mem_enable_outer_dim_regions(m)
        dll.iris_data_mem_create_region(byref(m), root_mem, c_int(region))
        return m
    else:
        (host, off, host_size, dev_size, elem_size, dim) = args
        m = iris_mem()
        c_host = host.ctypes.data_as(c_void_p)
        coff = (c_size_t * dim)(*off)
        chost_size = (c_size_t * dim)(*host_size)
        cdev_size = (c_size_t * dim)(*dev_size)
        size = host.nbytes
        dll.iris_data_mem_create_tile(byref(m), c_host, coff, chost_size, cdev_size, c_size_t(elem_size), c_int(dim))
        return m

def mem_reduce(mem, mode, type):
    return dll.iris_mem_reduce(mem, c_int(mode), c_int(type))

def mem_release(mem):
    return dll.iris_mem_release(mem)

def kernel_create(name):
    k = iris_kernel()
    c_name = c_void_p(0)
    if name != None:
        c_name = c_char_p(name) if sys.version_info[0] == 2 else c_char_p(bytes(name, 'ascii'))
    dll.iris_kernel_create(c_name, byref(k))
    return k

def kernel_setarg(kernel, idx, size, value):
    if type(value) == int: cvalue = byref(c_int(value))
    elif type(value) == float and size == 4: cvalue = byref(c_float(value))
    elif type(value) == float and size == 8: cvalue = byref(c_double(value))
    return dll.iris_kernel_setarg(kernel, c_int(idx), c_size_t(size), cvalue)

def kernel_setmem(kernel, idx, mem, mode):
    return dll.iris_kernel_setmem(kernel, c_int(idx), mem, c_size_t(mode))

def kernel_release(kernel):
    return dll.iris_kernel_release(kernel)

def graph_create(name = None):
    t = iris_graph()
    dll.iris_graph_create(byref(t))
    return t

def task_create(name = None):
    t = iris_task()
    c_name = c_void_p(0)
    if name != None:
        c_name = c_char_p(name) if sys.version_info[0] == 2 else c_char_p(bytes(name, 'ascii'))
    dll.iris_task_create_name(c_name, byref(t))
    return t

def task_depend(task, ntasks, tasks):
    return dll.iris_task_depend(task, c_int(ntasks), (iris_task * len(tasks))(*tasks))

def task_kernel(task, kernel, dim, off, gws, lws, params, params_info, hold_params):
    coff = (c_size_t * dim)(*off)
    cgws = (c_size_t * dim)(*gws)
    clws = (c_size_t * dim)(*lws)
    nparams = len(params)
    cparams = (c_void_p * nparams)()
    hold_params.clear()
    for i in range(nparams):
        if hasattr(params[i], 'handle') and isinstance(params[i].handle, iris_mem):
            cparams[i] = params[i].handle.class_obj
        elif isinstance(params[i], int) and params_info[i] == 4:
            p = byref(c_int(params[i]))
            hold_params.append(p)
            cparams[i] = cast(p, c_void_p)
        elif isinstance(params[i], float) and params_info[i] == 4:
            p = byref(c_float(params[i]))
            cparams[i] = cast(p, c_void_p)
            hold_params.append(p)
        elif isinstance(params[i], float) and params_info[i] == 8:
            p = byref(c_double(params[i]))
            cparams[i] = cast(p, c_void_p)
            hold_params.append(p)
        else: 
            print("error")

    cparams_info = (c_int * nparams)(*params_info)
    kernel_name = c_char_p(kernel) if sys.version_info[0] == 2 else c_char_p(bytes(kernel, 'ascii'))
    return dll.iris_task_kernel(task, kernel_name, c_int(dim), coff, cgws, clws, c_int(nparams), cparams, cparams_info)

def task_host(task, func, params):
  CWRAPPER = CFUNCTYPE(c_int, c_void_p, POINTER(c_int))
  wrapped_py_func = CWRAPPER(func)
  nparams = len(params)
  cparams = (c_void_p * nparams)(*params)
  return dll.iris_task_host(task, cast(wrapped_py_func, c_void_p), cparams)

def task_h2d(task, mem, off, size, host):
    return dll.iris_task_h2d(task, mem, c_size_t(off), c_size_t(size), host.ctypes.data_as(c_void_p))

def task_params_map(task, params_map_list):
    nparams = len(params_map_list)
    c_params_map_list = (c_int * nparams)(*params_map_list)
    return dll.iris_params_map(task, c_params_map_list)

def task_d2h(task, mem, off, size, host):
    return dll.iris_task_d2h(task, mem, c_size_t(off), c_size_t(size), host.ctypes.data_as(c_void_p))

def task_h2d_full(task, mem, host):
    return dll.iris_task_h2d_full(task, mem, host.ctypes.data_as(c_void_p))

def task_d2h_full(task, mem, host):
    return dll.iris_task_d2h_full(task, mem, host.ctypes.data_as(c_void_p))

def task_submit(task, device, sync):
    return dll.iris_task_submit(task, c_int(device), None, c_int(sync))

def task_wait(task):
    return dll.iris_task_wait(task)

def task_wait_all(ntask, tasks):
    return dll.iris_task_wait_all(c_int(ntask), (iris_task * len(tasks))(*tasks))

def task_add_subtask(task, subtask):
    return dll.iris_task_add_subtask(task, subtask)

def task_release(task):
    return dll.iris_task_release(task)

def task_release_mem(task, mem):
    return dll.iris_task_release_mem(task, mem)

def timer_now():
    d = c_double()
    dll.iris_timer_now(byref(d))
    return d.value

mem_handle_2_pobj = {}
class dmem_null:
  def __init__(self, size):
    m = iris_mem()
    c_host = c_void_p(0)
    dll.iris_data_mem_create(byref(m), c_host, c_size_t(size))
    self.handle = m
    mem_handle_2_pobj[self.handle.class_obj] = self
  def size(self):
    dll.iris_mem_get_size.restype = c_size_t
    return dll.iris_mem_get_size(self.handle)
  def update(self, host):
    c_host = host.ctypes.data_as(c_void_p)
    dll.iris_data_mem_update(self.handle, c_host)

class dmem:
  def __init__(self, *args):
    self.handle = dmem_create(args)
    self.type = IRIS_DMEM
    mem_handle_2_pobj[self.handle.class_obj] = self
  def is_reset(self):
    dll.iris_mem_is_reset.restype = c_int
    return dll.iris_mem_is_reset(self.handle)
  def uid(self):
    dll.iris_mem_get_uid.restype = c_ulong
    return dll.iris_mem_get_uid(self.handle)
  def size(self):
    dll.iris_mem_get_size.restype = c_size_t
    return dll.iris_mem_get_size(self.handle)
  def update(self, host):
    c_host = host.ctypes.data_as(c_void_p)
    dll.iris_data_mem_update(self.handle, c_host)
  def get_region_uids(self):
    n_regions = dll.call_ret(dll.iris_data_mem_n_regions(self.handle), np.int32)
    output = []
    for i in range(n_regions):
      output.append(dll.call_ret(dll.iris_data_mem_get_region_uid, np.uint32, self.handle, np.int32(i)))
    return output
class dmem_region:
  def __init__(self, *args):
    self.handle = dmem_create(args)
    self.type = IRIS_DMEM_REGION
    self.fetch_dmem()
  def fetch_dmem(self, handle=None):
    if handle == None:
        handle = self.handle
    mem_handle_2_pobj[handle.class_obj] = self
    dll.iris_get_dmem_for_region.restype = iris_mem
    mem_obj = dll.iris_get_dmem_for_region(handle)
    if mem_obj.class_obj not in mem_handle_2_pobj:
        pmem = dmem.__new__(dmem)
        pmem.type = IRIS_DMEM
        pmem.handle = mem_obj
        mem_handle_2_pobj[mem_obj.class_obj] = pmem
    pmem = mem_handle_2_pobj[mem_obj.class_obj]
    uid = pmem.uid()
    self.mem = pmem
  def is_reset(self):
    dll.iris_mem_is_reset.restype = c_int
    return dll.iris_mem_is_reset(self.handle)
  def uid(self):
    dll.iris_mem_get_uid.restype = c_ulong
    return dll.iris_mem_get_uid(self.handle)
  def size(self):
    dll.iris_mem_get_size.restype = c_size_t
    return dll.iris_mem_get_size(self.handle)
  def update(self, host):
    c_host = host.ctypes.data_as(c_void_p)
    dll.iris_data_mem_update(self.handle, c_host)
  def mem(self):
    return self.mem
class mem:
  def __init__(self, size):
    self.handle = mem_create(size)
    self.type = IRIS_MEM
    mem_handle_2_pobj[self.handle.class_obj] = self
  def is_reset(self):
    dll.iris_mem_is_reset.restype = c_int
    return dll.iris_mem_is_reset(self.handle)
  def uid(self):
    dll.iris_mem_get_uid.restype = c_ulong
    return dll.iris_mem_get_uid(self.handle)
  def size(self):
    dll.iris_mem_get_size.restype = c_size_t
    return dll.iris_mem_get_size(self.handle)
  def reduce(self, mode, type):
    mem_reduce(self.handle, mode, type)
  def release(self):
    mem_release(self.handle)

kernel_handle_2_pobj = {}
class kernel:
  def __init__(self, name):
    self.handle = kernel_create(name)
    kernel_handle_2_pobj[self.handle.class_obj] = self
  def name(self):
      dll.iris_kernel_get_name.restype = c_char_p
      kname = dll.iris_kernel_get_name(self.handle)
      return kname.decode('ascii')
  def uid(self):
      dll.iris_kernel_get_uid.restype = c_ulong
      return dll.iris_kernel_get_uid(self.handle)
  def setarg(self, idx, size, value):
    kernel_setarg(self.handle, idx, size, value)
  def setint(self, idx, value):
    kernel_setarg(self.handle, idx, 4, value)
  def setfloat(self, idx, value):
    kernel_setarg(self.handle, idx, 8, value)
  def setdouble(self, idx, value):
    kernel_setarg(self.handle, idx, 8, value)
  def setmem(self, idx, mem, mode):
    kernel_setmem(self.handle, idx, mem.handle, mode)
  def release(self):
    kernel_release(self.handle)

class size_t:
    def __init__(self, value):
        self.value = value

def get_scalar_conv(pobj):
    if isinstance(pobj, int) or type(pobj) == np.int32:
        return c_int(pobj)
    if type(pobj) == np.int64:
        return c_int64(pobj)
    if type(pobj) == np.double or type(pobj) == np.float64:
        return c_double(pobj)
    if isinstance(pobj, float) or type(pobj) == np.single or type(pobj) == np.float32:
        return c_float(pobj)
    if isinstance(pobj, bool):
        return c_bool(pobj)
    if type(pobj) == np.byte:
        return c_byte(pobj)
    if type(pobj) == np.ubyte:
        return c_ubyte(pobj)
    if type(pobj) == np.short:
        return c_short(pobj)
    if type(pobj) == np.ushort:
        return c_ushort(pobj)
    if type(pobj) == np.intc:
        return c_int(pobj)
    if type(pobj) == np.uintc:
        return c_uint(pobj)
    if type(pobj) == np.int_:
        return c_long(pobj)
    if type(pobj) == np.uint:
        return c_ulong(pobj)
    if type(pobj) == np.longlong:
        return c_longlong(pobj)
    if type(pobj) == np.ulonglong:
        return c_ulonglong(pobj)
    if type(pobj) == np.longdouble:
        return c_longdouble(pobj)
    if type(pobj) == str:
        cname = c_char_p(pobj) if sys.version_info[0] == 2 else c_char_p(bytes(pobj, 'ascii'))
        return cname
    return c_int(pobj)

def convert_obj_ctype(pobj):
    if type(pobj) != np.ndarray and pobj == None:
        cobj = c_void_p(0)
        return cobj, sizeof(cobj)
    elif np.isscalar(pobj):
        cobj = get_scalar_conv(pobj)
        return cobj, sizeof(cobj)
    elif type(pobj) == size_t:
        return c_size_t(pobj.value), sizeof(c_size_t)
    elif type(pobj) == np.ndarray and pobj.size > 1 and type(pobj[0]) == np.str_:
        str_list = []
        for s in str_list:
            c_str = c_char_p(s) if sys.version_info[0] == 2 else c_char_p(bytes(s, 'ascii'))
            s.append(c_str)
        s = np.array(str_list)
        cobj = s.ctypes.data_as(c_void_p)
        return cobj, s.nbytes
    elif type(pobj) == np.ndarray:
        cobj = pobj.ctypes.data_as(c_void_p)
        return cobj, pobj.nbytes
    elif type(pobj) == graph:
        return pobj.handle, sizeof(pobj.handle)
    elif type(pobj) == task:
        return pobj.handle, sizeof(pobj.handle)
    else:
        # Expecting user already defined its type
        c_host = pobj
        c_size = sizeof(pobj)
        return c_host, c_size
     
class kernel_arg:
    def __init__(self, is_mem, size, value, mem_obj, mem_off, mem_size, off, mode):
        self.is_mem = is_mem
        self.size = size
        self.value = None
        if not is_mem:
            self.value = cast(value, ctypes.POINTER(ctypes.c_ubyte * self.size))
        pmem = None
        if is_mem and mem_obj.class_obj not in mem_handle_2_pobj:
            dll.iris_mem_get_type.restype = c_int
            mem_type = dll.iris_mem_get_type(mem_obj)
            if mem_type == IRIS_DMEM_REGION:
                pmem = dmem_region.__new__(dmem_region)
                pmem.type = IRIS_DMEM_REGION
            elif mem_type == IRIS_DMEM:
                pmem = dmem.__new__(dmem)
                pmem.type = IRIS_DMEM
            else:
                pmem = mem.__new__(mem)
                pmem.type = IRIS_MEM
            pmem.handle = mem_obj
            mem_handle_2_pobj[mem_obj.class_obj] = pmem
            if pmem.type == IRIS_DMEM_REGION:
                pmem.fetch_dmem()
        self.mem = None
        if is_mem:
            self.mem = mem_handle_2_pobj[mem_obj.class_obj]
        self.mem_off = mem_off
        self.mem_size = mem_size
        self.off = off
        self.mode = mode
    def is_mode_r(self):
        return self.mode == iris_r
    def is_mode_rw(self):
        return self.mode == iris_rw
    def is_mode_w(self):
        return self.mode == iris_w
    def is_input(self):
        return self.is_mode_r() or self.is_mode_rw()
    def is_output(self):
        return self.is_mode_w() or self.is_mode_rw()
    def param_value(self, pack_type='f'):
        [ value ] = struct.unpack(pack_type, bytes(self.value.contents)) 
        return value
    def display(self):
        print(self.is_mem, self.size, self.value, self.mem, self.mem_off, self.mem_size, self.off, self.mode)

class cmd_kernel:
    def __init__(self, ckernel):
        self.ckernel = ckernel
    def name(self):
        dll.iris_kernel_get_name.restype = c_char_p
        kname = dll.iris_kernel_get_name(self.handle)
        return kname.decode('ascii')
    def get_kernel_nargs(self):
        dll.iris_cmd_kernel_get_nargs.restype = c_int
        return dll.iris_cmd_kernel_get_nargs(self.ckernel)
    def get_kernel_args(self):
        nargs = self.get_kernel_nargs()
        tasks = (POINTER(c_void_p) * nargs)()
        args = []
        dll.iris_cmd_kernel_get_arg_mode.restype = c_int
        dll.iris_cmd_kernel_get_arg_size.restype = c_size_t
        dll.iris_cmd_kernel_get_arg_mem_off.restype = c_size_t
        dll.iris_cmd_kernel_get_arg_mem_size.restype = c_size_t
        dll.iris_cmd_kernel_get_arg_off.restype = c_size_t
        dll.iris_cmd_kernel_get_arg_mem.restype = iris_mem
        dll.iris_cmd_kernel_get_arg_value.restype = POINTER(c_ubyte)
        for i in range(nargs):
            is_mem = dll.iris_cmd_kernel_get_arg_is_mem(self.ckernel, i)
            mode = dll.iris_cmd_kernel_get_arg_mode(self.ckernel, i)
            size = dll.iris_cmd_kernel_get_arg_size(self.ckernel, i)
            mem_off = dll.iris_cmd_kernel_get_arg_mem_off(self.ckernel, i)
            mem_size = dll.iris_cmd_kernel_get_arg_mem_size(self.ckernel, i)
            off = dll.iris_cmd_kernel_get_arg_off(self.ckernel, i)
            mem_obj = dll.iris_cmd_kernel_get_arg_mem(self.ckernel, i)
            value = dll.iris_cmd_kernel_get_arg_value(self.ckernel, i)
            args.append(kernel_arg(is_mem, size, value, mem_obj, mem_off, mem_size, off, mode))
        return nargs, args
task_handle_2_pobj = {}
class task:
    def __init__(self, *args):
        if len(args) == 0:
            self.params = []
            self.handle = task_create()
            task_handle_2_pobj[self.handle.class_obj] = self
            return 
        (kernel, dim, off, gws, lws, params_list) = args
        coff = (c_size_t * dim)(*off)
        cgws = (c_size_t * dim)(*gws)
        clws = (c_size_t * dim)(*lws)
        nparams = len(params_list)
        self.params = []
        cparams = (c_void_p * nparams)()
        self.handle = task_create(kernel)
        task_handle_2_pobj[self.handle.class_obj] = self
        params_info = []
        flush_objs = []
        for i, pobj in enumerate(params_list):
            if type(pobj) == tuple:
                if hasattr(pobj[0], 'handle') and isinstance(pobj[0].handle, iris_mem):
                    self.params.append(pobj[0])
                    cparams[i] = pobj[0].handle.class_obj
                    params_info.append(pobj[1])
                    if len(pobj) == 3 and pobj[2] == iris_flush:
                        flush_objs.append(cparams[i])
                else:
                    # (NUMPY Object, modes of use, <optional flush>)
                    # Should be numpy array
                    host = pobj[0]
                    dobj = dmem(host)
                    self.params.append(dobj)
                    cparams[i] = dobj.handle.class_obj
                    params_info.append(pobj[1])
                    if len(pobj) == 3 and pobj[2] == iris_flush:
                        flush_objs.append(cparams[i])
            elif pobj == None:
                p = c_void_p(0)
                self.params.append(p)
                cparams[i] = cast(p, c_void_p)
                params_info.append(sizeof(p))
            elif np.isscalar(pobj):
                cobj = get_scalar_conv(pobj)
                p = byref(cobj)
                self.params.append(p)
                cparams[i] = cast(p, c_void_p)
                params_info.append(sizeof(cobj))
            elif type(pobj) == size_t:
                cobj = c_size_t(pobj.value)
                p = byref(cobj)
                self.params.append(p)
                cparams[i] = cast(p, c_void_p)
                params_info.append(sizeof(cobj))
            else: 
                print("error")
        cparams_info = (c_int * nparams)(*params_info)
        kernel_name = c_char_p(kernel) if sys.version_info[0] == 2 else c_char_p(bytes(kernel, 'ascii'))
        dll.iris_task_kernel(self.handle, kernel_name, c_int(dim), coff, cgws, clws, c_int(nparams), cparams, cparams_info)
        for pobj in flush_objs:
            dll.iris_task_dmem_flush_out(self.handle, pobj);
    def name(self):
        dll.iris_task_get_name.restype = c_char_p
        kname = dll.iris_task_get_name(self.handle)
        return kname.decode('ascii')

    def set_order(self, order):
        dll.call(dll.iris_task_kernel_dmem_fetch_order, self.handle, order)

    def set_name(self, name):
        c_name = c_char_p(name) if sys.version_info[0] == 2 else c_char_p(bytes(name, 'ascii'))
        dll.iris_task_set_name(self.handle, c_name)
    def set_device(self, device):
        dll.iris_task_set_policy(self.handle, convert_obj_ctype(device)[0])
    def uid(self):
        dll.iris_task_get_uid.restype = c_ulong
        return dll.iris_task_get_uid(self.handle)
    def get_kernel_args(self):
        ckernel = self.get_cmd_kernel()
        if ckernel == None:
            return 0, []
        return ckernel.get_kernel_args()

    def get_input_mems(self):
        ckernel = self.get_cmd_kernel()
        if ckernel == None:
            return []
        n_args, args = ckernel.get_kernel_args()
        inputs = []
        for each_arg in args:
            if each_arg.mem != None and each_arg.is_input():
                inputs.append(each_arg.mem)
        return inputs

    def get_output_mems(self):
        ckernel = self.get_cmd_kernel()
        if ckernel == None:
            return []
        n_args, args = ckernel.get_kernel_args()
        outputs = []
        for each_arg in args:
            if each_arg.mem != None and each_arg.is_output():
                outputs.append(each_arg.mem)
        return outputs

    def get_cmd_kernel(self):
        dll.iris_task_is_cmd_kernel_exists.restype = c_int
        is_exists = dll.iris_task_is_cmd_kernel_exists(self.handle)
        if is_exists == 0:
            return None
        dll.iris_task_get_cmd_kernel.restype = POINTER(c_void_p)
        ckernel = dll.iris_task_get_cmd_kernel(self.handle)
        return cmd_kernel(ckernel)

    def get_kernel(self):
        dll.iris_task_get_kernel.restype = iris_kernel
        ckernel = dll.iris_task_get_kernel(self.handle)
        if ckernel.class_obj not in kernel_handle_2_pobj:
            pkernel = kernel.__new__(kernel)
            pkernel.handle = ckernel
            kernel_handle_2_pobj[ckernel.class_obj] = pkernel
        return kernel_handle_2_pobj[ckernel.class_obj]

    def get_ndepends(self):
        dll.iris_task_get_dependency_count.restype = c_int
        ntasks = dll.iris_task_get_dependency_count(self.handle)
        return ntasks

    def get_depends(self):
        dll.iris_task_get_dependency_count.restype = c_int
        ntasks = dll.iris_task_get_dependency_count(self.handle)
        tasks = (iris_task * ntasks)()
        dll.iris_task_get_dependencies.argtypes = [iris_task, POINTER(iris_task)]
        dll.iris_task_get_dependencies.restype = None
        dll.iris_task_get_dependencies(self.handle, tasks)
        ptasks = convert_ctasks(tasks)
        return ntasks, ptasks
        
    def depends(self, tasks_list):
        if type(tasks_list) != list:
            tasks_list = [ tasks_list ]
        dep_list = [t.handle for t in tasks_list]
        task_depend(self.handle, len(tasks_list), dep_list)
    def h2d(self, mem, off, size, host):
        task_h2d(self.handle, mem.handle, off, size, host)
    def params_map(self, params_map_list):
        task_params_map(self.handle, params_map_list)
    def d2h(self, mem, off, size, host):
        task_d2h(self.handle, mem.handle, off, size, host)
    def h2d_full(self, mem, host):
        task_h2d_full(self.handle, mem.handle, host)
    def d2h_full(self, mem, host):
        task_d2h_full(self.handle, mem.handle, host)
    def kernel(self, kernel, dim, off, gws, lws, params, params_info):
        task_kernel(self.handle, kernel, dim, off, gws, lws, params, params_info, self.params)
    def host(self, func, params):
        task_host(self.handle, func, params)
    def submit(self, device, sync = 1):
        task_submit(self.handle, device, sync)
    def task_name(self):
        dll.iris_task_get_name(self.handle)
    def kernel_name(self):
        dll.iris_task_get_kernel_name(self.handle)
    def kernel_params(self):
        dll.iris_task_get_kernel_params(self.handle)
    def inputs(self):
        dll.iris_task_get_inputs(self.handle)
    def outputs(self):
        dll.iris_task_get_inputs(self.handle)

def convert_ctasks(c_tasks):
    ptasks = []
    for c_task in c_tasks:
        if c_task.class_obj not in task_handle_2_pobj:
            p_task = task.__new__(task)
            p_task.params = []
            p_task.handle = c_task
            task_handle_2_pobj[c_task.class_obj] = p_task
        ptasks.append(task_handle_2_pobj[c_task.class_obj])
    return ptasks
class graph:
    def __init__(self, tasks=[], device=iris_default, opt=None):
        self.handle = graph_create()
        for each_task in tasks:
            self.add_task(each_task, device, opt)
    def retain(self):
        dll.iris_graph_retain(self.handle)
    def release(self):
        dll.iris_graph_release(self.handle)
    def free(self):
        dll.iris_graph_free(self.handle)
    def wait(self):
        dll.iris_graph_wait(self.handle)
    def submit(self, device=iris_default, sync=1, order=None):
        if type(order) == np.ndarray or order != None:
            order_np = order
            if type(order) != np.ndarray:
                order_np = np.int32(order)
            conv_order_np = convert_obj_ctype(order_np)
            dll.iris_graph_submit_with_order(self.handle, conv_order_np[0], c_int(device), c_int(sync))
        else:
            dll.iris_graph_submit(self.handle, c_int(device), c_int(sync))
    def submit_time(self, device=iris_default, sync=1, order=None):
        #stime = np.double(0.0)
        #cobj = byref(c_double(stime))
        stime = np.zeros((1), np.double)
        cobj = stime.ctypes.data_as(c_void_p)
        if type(order) == np.ndarray or order != None:
            order_np = order
            if type(order) != np.ndarray:
                order_np = np.int32(order)
            conv_order_np = convert_obj_ctype(order_np)
            dll.iris_graph_submit_with_order_and_time(self.handle, conv_order_np[0], cobj, c_int(device), c_int(sync))
        else:
            dll.iris_graph_submit_with_time(self.handle, cobj, c_int(device), c_int(sync))
        return stime[0]
    def add_task(self, task, device=iris_default, opt=None):
        c_opt = c_void_p(0)
        if opt != None:
           c_opt = c_char_p(opt) if sys.version_info[0] == 2 else c_char_p(bytes(opt, 'ascii'))
        dll.iris_graph_task(self.handle, task.handle, c_int(device), c_opt)

    def reset_memories(self):
        dll.iris_graph_reset_memories(self.handle)

    def enable_mem_profiling(self):
        dll.iris_graph_enable_mem_profiling(self.handle)

    def set_order(self, order):
        dll.call(dll.iris_graph_tasks_order, self.handle, order)

    def get_tasks(self):
        dll.iris_graph_tasks_count.restype = c_int
        ntasks = dll.iris_graph_tasks_count(self.handle)
        tasks = (iris_task * ntasks)()
        dll.iris_graph_get_tasks.argtypes = [iris_graph, POINTER(iris_task)]
        dll.iris_graph_get_tasks.restype = None
        dll.iris_graph_get_tasks(self.handle, tasks)
        ptasks = convert_ctasks(tasks)
        return ntasks, ptasks

    def get_task_names(self):
        ntasks, tasks = self.get_tasks()
        task_names = [ task.name() for task in tasks]
        task_names[0] = 'end_task'
        task_names.insert(0, 'start_task')
        return task_names
    def get_task_uids(self):
        ntasks, tasks = self.get_tasks()
        task_uids = [0] + [ task.uid() for task in tasks]
        return task_uids

    def tasks_execution_schedule(self, kernel_profile=False):
        # Order should not be changed
        ptr = dll.call_ret(dll.iris_get_graph_tasks_execution_schedule, POINTER(TaskProfile), self.handle, np.int32(kernel_profile))
        total = dll.call_ret(dll.iris_get_graph_tasks_execution_schedule_count, c_size_t, self.handle)
        ptr_list = [ ptr[i] for i in range(total) ]
        return ptr_list

    def mems_execution_schedule(self):
        # Order should not be changed
        ptr = dll.call_ret(dll.iris_get_graph_dataobjects_execution_schedule, POINTER(DataObjectProfile), self.handle)
        total = dll.call_ret(dll.iris_get_graph_dataobjects_execution_schedule_count, c_size_t, self.handle)
        ptr_list = [ ptr[i] for i in range(total) ]
        return ptr_list

    def get_dependency_matrix(self, pdf=False):
        ntasks, tasks = self.get_tasks()
        SIZE = ntasks+1
        dep_graph, dep_graph_2d_ptr = dll.alloc_int8((SIZE,SIZE))
        dll.call_ret_ptr(dll.iris_get_graph_dependency_adj_matrix, self.handle, dep_graph)
        print("Dependency matrix", dep_graph)
        if pdf:
            task_names = self.get_task_names()
            task_uids =  self.get_task_uids()
            import pandas as pd
            df = pd.DataFrame(dep_graph, columns=task_names)
            df['task_name'] = task_names
            df['task_uid'] = task_uids
            df['parent_count'] = dep_graph.sum(axis=1)
            df['child_count'] = dep_graph.sum(axis=0)
            return df
        return dep_graph
    def get_3d_cost_communication_matrix(self, pdf=False):
        ntasks, tasks = self.get_tasks()
        task_index_hash = {}
        task_input_mems = []
        task_output_mems = []
        mem_index_hash = {}
        mem_regions_2_dmem_hash = {}
        mem_dmem_2_regions = {}
        mem_serial_index_hash = {}
        for index, each_task in enumerate(tasks):
            task_index_hash[each_task.uid()] = index
            inputs = each_task.get_input_mems()
            outputs = each_task.get_output_mems()
            task_input_mems.append([])
            task_output_mems.append([])
            for inp in inputs:
                if inp.uid() not in mem_index_hash:
                    mem_index_hash[inp.uid()] = {'mem': inp, 'inputs':[], 'outputs':[]}
                mem_index_hash[inp.uid()]['inputs'].append(each_task.uid())
                task_input_mems[index].append(inp.uid())
                if inp.uid() not in mem_serial_index_hash:
                    mem_serial_index_hash[inp.uid()] = len(mem_serial_index_hash)
            for outp in outputs:
                if outp.uid() not in mem_index_hash:
                    mem_index_hash[outp.uid()] = {'mem': outp, 'inputs':[], 'outputs':[]}
                mem_index_hash[outp.uid()]['outputs'].append(each_task.uid())
                task_output_mems[index].append(outp.uid())
                if outp.uid() not in mem_serial_index_hash:
                    mem_serial_index_hash[outp.uid()] = len(mem_serial_index_hash)
                if outp.type == IRIS_DMEM_REGION:
                    if outp.uid() not in mem_regions_2_dmem_hash:
                        mem_regions_2_dmem_hash[outp.uid()] = outp.mem.uid()
                    if outp.mem.uid() not in mem_dmem_2_regions:
                        mem_dmem_2_regions[outp.mem.uid()] = []
                    mem_dmem_2_regions[outp.mem.uid()].append(outp.uid())
        task_outputs = np.zeros(len(tasks))
        for index, each_task in enumerate(tasks):
            n_depends, dep_tasks = each_task.get_depends()
            for d_task in dep_tasks:
                d_index = task_index_hash[d_task.uid()]
                task_outputs[d_index] += 1
        dep_graph = np.zeros((len(tasks)+1, len(tasks)+1, len(mem_index_hash)), np.int64)
        for index, each_task in enumerate(tasks):
            lst1 = task_input_mems[index]
            n_depends, dep_tasks = each_task.get_depends()
            if index == 0:
                # End Task
                dep_tasks = []
                for d_index, n_outs in enumerate(task_outputs):
                    if n_outs == 1:
                        dep_tasks.append(tasks[d_index])
            added_mems = []
            for d_task in dep_tasks:
                d_index = task_index_hash[d_task.uid()]
                lst2 = task_output_mems[d_index]
                mem_list = lst2
                if len(lst1) > 0:
                    mem_list = list(set(lst1) & set(lst2))
                added_mems = mem_list
                for mem_index in mem_list:
                    size = mem_index_hash[mem_index]['mem'].size()
                    mem_serial_index = mem_serial_index_hash[mem_index]
                    dep_graph[index+1][d_index+1][mem_serial_index] += size
                for mem_index in lst2:
                    if mem_index in mem_regions_2_dmem_hash:
                        dmem_index = mem_regions_2_dmem_hash[mem_index]
                        if dmem_index in lst1:
                            size = mem_index_hash[mem_index]['mem'].size()
                            mem_serial_index = mem_serial_index_hash[mem_index]
                            dep_graph[index+1][d_index+1][mem_serial_index] += size
                            added_mems.append(dmem_index)
            for mem_index in lst1:
                if mem_index not in added_mems:
                    mem_obj = mem_index_hash[mem_index]['mem']
                    if mem_index in mem_regions_2_dmem_hash:
                        dmem_index = mem_regions_2_dmem_hash[mem_index]
                        dmem_obj = mem_index_hash[dmem_index]['mem']
                        if not dmem_obj.is_reset():
                            size = mem_obj.size()
                            mem_serial_index = mem_serial_index_hash[mem_index]
                            dep_graph[index+1][0][mem_serial_index] += size
                    elif not mem_obj.is_reset():
                        size = mem_obj.size()
                        mem_serial_index = mem_serial_index_hash[mem_index]
                        dep_graph[index+1][0][mem_serial_index] += size
        return dep_graph

    def get_2d_cost_communication_matrix(self, pdf=False):
        ntasks, tasks = self.get_tasks()
        SIZE = ntasks+1
        comm_2d, comm_2d_ptr = dll.alloc_size_t((SIZE,SIZE))
        dll.call_ret_ptr(dll.iris_get_graph_2d_comm_adj_matrix, self.handle, comm_2d)
        print("Communication cost matrix", comm_2d)
        if pdf:
            task_names = self.get_task_names()
            task_uids =  self.get_task_uids()
            import pandas as pd
            df = pd.DataFrame(comm_2d, columns=task_names)
            df['task_name'] = task_names
            df['task_uid'] = task_uids
            df['parent_count'] = comm_2d.sum(axis=1)
            df['child_count'] = comm_2d.sum(axis=0)
            return df
        return comm_2d

    def get_3d_cost_comm_time(self, iterations=1, pin_memory=True):
        print("Extracting 3d communication cost for each object")
        n_mems = dll.call_ret(dll.iris_count_mems, c_size_t, self.handle)
        ndevs = device_count()+1
        time_data = np.zeros(n_mems*ndevs*ndevs, np.double)
        mem_ids = np.zeros(n_mems, np.int32)
        dll.call(dll.iris_get_graph_3d_comm_time, self.handle, time_data, mem_ids, np.int32(iterations), np.int32(pin_memory))
        return time_data.reshape((n_mems, ndevs, ndevs)), mem_ids 

    def get_3d_cost_comm_data(self):
        print("Extracting communication data 3d list of (task from id, task to id, mem_id, size)")
        dll.call(dll.iris_get_graph_3d_comm_data, self.handle)
        total = dll.call_ret(dll.iris_get_graph_3d_comm_data_size, c_size_t, self.handle)
        ptr = dll.call_ret(dll.iris_get_graph_3d_comm_data_ptr, POINTER(CommData3D), self.handle)
        ptr_list = [ ptr[i] for i in range(total) ]
        return ptr_list

    def calibrate_compute_cost_adj_matrix(self, pdf=False, only_device_type=False):
        print("Calibrating computation cost for each task and for each processor type")
        ntasks, tasks = self.get_tasks()
        ndevs = device_count()
        dev_names = all_device_info()
        SIZE = ntasks+1
        cols = ndevs
        col_names = dev_names
        if only_device_type:
            cols = platform_count()
            col_names = all_platform_info()
        comp_2d, comp_2d_ptr = dll.alloc_double((SIZE, cols))
        if only_device_type:
            dll.call_ret_ptr(dll.iris_calibrate_compute_cost_adj_matrix_only_for_types, self.handle, comp_2d)
        else:
            dll.call_ret_ptr(dll.iris_calibrate_compute_cost_adj_matrix, self.handle, comp_2d)
        print("Computation cost matrix", comp_2d)
        if pdf:
            task_names = self.get_task_names()
            task_uids =  self.get_task_uids()
            import pandas as pd
            df = pd.DataFrame(comp_2d, columns=col_names)
            df['task_name'] = task_names
            df['task_uid'] = task_uids
            return df
        return comp_2d

