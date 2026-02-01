#!/usr/bin/env python3

"""
IRIS Python interface
"""
__author__ = "Narasinga Rao Miniskar"
__copyright__ = "Copyright (c) 2020-2023, Oak Ridge National Laboratory (ORNL) Programming Systems Research Group. All rights reserved."
__license__ = "UT Battelle Open Source"
__version__ = "1.0"

import ctypes
from ctypes import *
import sys
import numpy as np
import pdb
import os
import struct

path = os.path.dirname(os.path.abspath(__file__))
if 'IRIS' not in os.environ:
    os.environ['IRIS'] = path

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


iris_default_libname = "libiris.so"
if sys.platform.startswith("linux"):
    iris_default_libname = "libiris.so"
elif sys.platform == "darwin":
    iris_default_libname = "libiris.dylib"

class IRIS(library):
    def __init__(self, libname=iris_default_libname):
        so_file = libname
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(current_dir, 'lib64', libname)):
            so_file = os.path.join(current_dir, 'lib64', libname)
        elif os.path.exists(os.path.join(current_dir, 'lib', libname)):
            so_file = os.path.join(current_dir, 'lib', libname)
        # Do not error on import when Iris is not built. This is needed for Read the Docs to create the API documentation.
        try:
            super(IRIS, self).__init__(so_file)
        except OSError as e:
            print(e)
            print(f'{libname} not found. Please install IRIS library.', file=sys.stderr)

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
iris_sdq        =       (1 << 24)
iris_ftf        =       (1 << 25)
iris_ocl        =       (1 << 26)
iris_block_cycle=       (1 << 27)
iris_custom     =       (1 << 28)
iris_qpu        =       (1 << 30)
iris_python_policy =    (1 << 31)
iris_all        =       0x7FFFFFFF

iris_python_native     = 0

iris_r          =   -1
iris_w          =   -2
iris_rw         =   -3
iris_flush      =   -4

iris_unknown          =  (0 << 16)
iris_int              =  (1 << 16)
iris_uint             =  (2 << 16)
iris_float            =  (3 << 16)
iris_double           =  (4 << 16)
iris_char             =  (5 << 16)
iris_int8             =  (6 << 16)
iris_uint8            =  (7 << 16)
iris_int16            =  (8 << 16)
iris_uint16           =  (9 << 16)
iris_int32            =  (10 << 16)
iris_uint32           =  (11 << 16)
iris_int64            =  (12 << 16)
iris_uint64           =  (13 << 16)
iris_long             =  (14 << 16)
iris_unsigned_long    =  (15 << 16)
iris_bool             =  (16 << 16)
iris_custom_type      =  (17 << 16)
iris_pointer          =  (0x40 << 16)

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

DMEM_MAX_DIM = 6
IRIS_MEM = 0x1
IRIS_DMEM = 0x2
IRIS_DMEM_REGION = 0x4

IRIS_BASE_TYPE_MASK = ~iris_pointer

CQireeResultRecord = np.dtype([
    ('key',   np.uint64),
    ('count', np.uint64)
])

CProfileRecord = np.dtype([
    ('start_time',   np.float64),
    ('end_time',     np.float64),
    ('type',         np.int32),
    ('stream',       np.int32),
    ('id',           np.int64)
])

class IrisParam(ctypes.Structure):
    _fields_ = [
        ("data",        c_void_p),          # void* data
        ("off",         POINTER(c_size_t)), # size_t* off
        ("tile_sizes",  POINTER(c_size_t)), # size_t* tile_sizes
        ("full_sizes",  POINTER(c_size_t)), # size_t* full_sizes
        ("element_size", c_size_t),         # element size in bytes
        ("count",       c_size_t),          # number of elements
        ("data_type",   c_int32),           # iris_* code
        ("is_pointer",  c_int32),           # 0 = scalar, 1 = pointer/array
        ("n_dims",      c_int32),           # rank of array
        ("mem_id",      c_ulong),           # user-defined
    ]
    def get_arr(self, ptr):
        nd = int(self.n_dims)

        if not ptr or nd == 0:
            return None
        return [int(ptr[i]) for i in range(nd)]

    def full_sizes_arr(self):
        return self.get_arr(self.full_sizes)

    def tile_sizes_arr(self):
        return self.get_arr(self.tile_sizes)

    def off_arr(self):
        return self.get_arr(self.off)

    def __repr__(self):
        vals = {}
        vals["data"]         = hex(self.data) if self.data else None
        vals["element_size"] = int(self.element_size)
        vals["count"]        = int(self.count)
        vals["data_type"]    = int(self.data_type)
        vals["is_pointer"]   = int(self.is_pointer)
        vals["n_dims"]       = int(self.n_dims)
        vals["mem_id"]       = int(self.mem_id)

        vals["off"]        = self.get_arr(self.off)
        vals["tile_sizes"] = self.get_arr(self.tile_sizes)
        vals["full_sizes"] = self.get_arr(self.full_sizes)

        # pretty-print
        inner = ", ".join(f"{k}={v}" for k, v in vals.items())
        return f"IrisParam({inner})"


class IrisSpec(ctypes.Structure):
    _fields_ = [
        ("devno",      c_int32),
        ("local_dev_id",    c_int32),            
        ("ordinal",     c_int32),
        ("model",      c_int32),            
        ("n_streams",  c_int32),
        ("stream_index",c_int32),
        ("is_async",   c_int32),
        ("dim",        c_int32),            # dimensionality for gws/grid/block
        ("task_id",    c_ulong),
        ("cmd_id",     c_ulong),
        ("ctx",        c_void_p),           # void* ctx
        ("gws",        POINTER(c_size_t)),  # size_t* gws
        ("off",        POINTER(c_size_t)),  # size_t* off
        ("grid",       POINTER(c_size_t)),  # size_t* grid
        ("block",      POINTER(c_size_t)),  # size_t* block
        ("streams",    POINTER(c_void_p)),  # void** streams
        ("shmem_size", c_size_t),
    ]

TYPE_TO_CTYPE = {
    iris_int:           c_int32,
    iris_int32:         c_int32,
    iris_uint:          c_uint32,
    iris_uint32:        c_uint32,
    iris_float:         ctypes.c_float,
    iris_double:        ctypes.c_double,
    iris_char:          ctypes.c_char,
    iris_int8:          c_int8,
    iris_uint8:         c_uint8,
    iris_int16:         c_int16,
    iris_uint16:        c_uint16,
    iris_int64:         c_int64,
    iris_uint64:        c_uint64,
    iris_long:          c_long,
    iris_unsigned_long: c_ulong,
    iris_bool:          c_uint8,  # or ctypes.c_bool
}

TYPE_TO_DTYPE = {
    iris_int:           np.int32,
    iris_int32:         np.int32,
    iris_uint:          np.uint32,
    iris_uint32:        np.uint32,
    iris_float:         np.float32,
    iris_double:        np.float64,
    iris_char:          np.byte,
    iris_int8:          np.int8,
    iris_uint8:         np.uint8,
    iris_int16:         np.int16,
    iris_uint16:        np.uint16,
    iris_int64:         np.int64,
    iris_uint64:        np.uint64,
    iris_long:          np.int64,   # adjust if your platform differs
    iris_unsigned_long: np.uint64,
    iris_bool:          np.bool_,
}

def iris_base_type(t: int) -> int:
    """Strip pointer bits from data_type."""
    return t & IRIS_BASE_TYPE_MASK

def is_custom_dtype(arr: np.ndarray) -> bool:
    return arr.fields is not None  # True for structured dtypes

def get_iris_type(np_dtype):
    if np_dtype == np.float64:
        return iris_double
    if np_dtype == np.float32:
        return iris_float
    if np_dtype == np.int8:
        return iris_int8
    if np_dtype == np.int16:
        return iris_int16
    if np_dtype == np.int32:
        return iris_int32
    if np_dtype == np.int64:
        return iris_int64
    if np_dtype == np.uint8:
        return iris_uint8
    if np_dtype == np.uint16:
        return iris_uint16
    if np_dtype == np.uint32:
        return iris_uint32
    if np_dtype == np.uint64:
        return iris_uint64
    if is_custom_dtype(np_dtype):
        return iris_custom_type
    if np_dtype.itemsize == 4:
        return iris_uint32
    if np_dtype.itemsize == 8:
        return iris_uint64
    return iris_double
class iris_kernel(Structure):
    _fields_ = [("class_obj", c_void_p), ("uid", c_ulong)]

class iris_command(Structure):
    _fields_ = [("class_obj", c_void_p), ("uid", c_ulong)]

class iris_mem(Structure):
    _fields_ = [("class_obj", c_void_p), ("uid", c_ulong)]

class iris_task(Structure):
    _fields_ = [("class_obj", c_void_p), ("uid", c_ulong)]

class iris_graph(Structure):
    _fields_ = [("class_obj", c_void_p), ("uid", c_ulong)]

# ---------------------------------------------------------------------------
#  Helpers: element count and shape from IrisParam
# ---------------------------------------------------------------------------

def param_nelems(p: IrisParam) -> int:
    """
    Number of elements represented by this IrisParam.
    Prefer p.count if set; otherwise, derive from full_sizes.
    """
    if p.count:
        return int(p.count)

    # Fallback: scalar
    if not p.is_pointer or p.n_dims == 0:
        return 1

    # Fallback: product of full_sizes
    if not p.full_sizes:
        return 1

    n = 1
    for i in range(int(p.n_dims)):
        n *= int(p.full_sizes[i])
    return n


def param_shape(p: IrisParam):
    """
    Shape tuple for a param.
    Uses full_sizes[0..n_dims-1] if available, else () or (count,).
    """
    if not p.is_pointer or p.n_dims == 0:
        return ()

    if p.full_sizes and p.n_dims > 0:
        return tuple(int(p.full_sizes[i]) for i in range(int(p.n_dims)))

    # No full_sizes pointer: treat as 1D of length count (if non-zero)
    nelems = param_nelems(p)
    return (nelems,) if nelems > 0 else ()


# ---------------------------------------------------------------------------
#  IrisParam -> Python (scalar or NumPy array view)
# ---------------------------------------------------------------------------

def param_to_python(p: IrisParam):
    """
    Convert an IrisParam into a Python object:
      - scalar -> Python scalar
      - array  -> NumPy array view on C memory
    """
    base = iris_base_type(p.data_type)

    # Unknown / custom type: just pass through pointer
    if base not in TYPE_TO_CTYPE:
        if p.mem_id in mem_handle_2_pobj:
            mem = mem_handle_2_pobj[p.mem_id]
            dtype = mem[0].dtype
            byte_ptr = ctypes.cast(p.data, ctypes.POINTER(ctypes.c_ubyte))
            full_sizes = p.full_sizes_arr()
            shape = tuple(int(s) for s in full_sizes)
            n_elems = int(np.prod(shape))
            total_bytes = n_elems * dtype.itemsize
            byte_array = np.ctypeslib.as_array(byte_ptr, shape=(total_bytes,))
            arr = byte_array.view(dtype).reshape(shape)
            return arr
        return p.data

    c_scalar_type = TYPE_TO_CTYPE[base]
    dtype         = TYPE_TO_DTYPE[base]

    # Optionally, sanity-check element_size vs ctypes size
    if p.element_size and ctypes.sizeof(c_scalar_type) != p.element_size:
        # You *can* assert or log here if you want stricter checking.
        # For now, we just trust the mapping.
        pass

    nelems = param_nelems(p)

    # Scalar case: not pointer/array or logically single element
    if (not p.is_pointer or p.n_dims == 0) or nelems == 1:
        ptr = ctypes.cast(p.data, ctypes.POINTER(c_scalar_type))
        return ptr[0]

    # Array case: build NumPy view over C memory
    ptr = ctypes.cast(p.data, ctypes.POINTER(c_scalar_type))
    flat = np.ctypeslib.as_array(ptr, shape=(nelems,))

    shape = param_shape(p)
    if shape and int(np.prod(shape)) == nelems:
        return flat.reshape(shape)
    else:
        return flat  # fallback: 1D


# ---------------------------------------------------------------------------
#  Python -> IrisParam (write-back into C-owned memory)
# ---------------------------------------------------------------------------

def python_to_param(value, p: IrisParam):
    """
    Write a Python scalar/array back into the memory pointed by IrisParam.
    Assumes C side has already allocated data with correct element_size/count.
    """
    base = iris_base_type(p.data_type)
    if base not in TYPE_TO_CTYPE:
        raise TypeError(f"Unsupported IRIS type for write-back: {hex(base)}")

    c_scalar_type = TYPE_TO_CTYPE[base]
    dtype         = TYPE_TO_DTYPE[base]

    nelems = param_nelems(p)

    # scalar
    if (not p.is_pointer or p.n_dims == 0) or nelems == 1:
        ptr      = ctypes.cast(p.data, ctypes.POINTER(c_scalar_type))
        ptr[0]   = c_scalar_type(value)
        return

    # array
    ptr  = ctypes.cast(p.data, ctypes.POINTER(c_scalar_type))
    view = np.ctypeslib.as_array(ptr, shape=(nelems,))

    arr = np.asarray(value, dtype=dtype)
    if arr.size != nelems:
        raise ValueError(
            f"Size mismatch in write-back: expected {nelems}, got {arr.size}"
        )

    view[...] = arr.reshape(view.shape)


# ---------------------------------------------------------------------------
#  IrisSpec -> Python
# ---------------------------------------------------------------------------

def spec_to_python(spec_ptr: POINTER(IrisSpec)):
    """
    Convert IrisSpec* into a Python dict with nice fields.
    """
    if not spec_ptr:
        return None

    s = spec_ptr.contents

    # Streams as a Python list of void* (opaque)
    streams = []
    if s.streams and s.n_streams > 0:
        for i in range(int(s.n_streams)):
            streams.append(s.streams[i])

    dim = int(s.dim)

    def read_size_array(ptr):
        if not ptr or dim <= 0:
            return None
        return [int(ptr[i]) for i in range(dim)]

    return {
        "n_streams": int(s.n_streams),
        "devno":     int(s.devno),
        "dev_id":    int(s.dev_id),
        "is_async":  int(s.is_async),
        "task_id":   int(s.task_id),
        "cmd_id":    int(s.cmd_id),
        "streams":   streams,    # list of c_void_p
        "ctx_ptr":   s.ctx,      # opaque pointer
        "dim":       dim,
        "gws":       read_size_array(s.gws),
        "grid":      read_size_array(s.grid),
        "block":     read_size_array(s.block),
        "shmem_size": int(s.shmem_size),
        "raw":       s,          # original IrisSpec ctypes struct
    }

# Callback type: void (*)(IrisParam* params, int32_t n, IrisSpec* spec)
__PY_IRIS_CB = CFUNCTYPE(None, c_void_p, POINTER(IrisSpec), POINTER(IrisParam), c_int32)

@__PY_IRIS_CB
def __py_iris_callback_kernel(kname_ptr, spec_ptr, params_ptr, n_params):
    # Convert IrisParam[] → Python arguments
    params  = [params_ptr[i] for i in range(int(n_params))]
    py_args = [param_to_python(p) for p in params]

    # Convert IrisSpec → Python dict
    spec_py = spec_to_python(spec_ptr)

    # Call the user function
    pyobj_p = ctypes.cast(kname_ptr, ctypes.POINTER(py_object))
    py_func = pyobj_p.contents.value
    result = py_func(spec_py, *py_args)

    # Write-back semantics (optional)
    if result is None:
        return

    # Single return value -> write into first param
    if not isinstance(result, (tuple, list)):
        try:
            python_to_param(result, params[0])
        except Exception:
            pass
        return

    # Multiple return values -> write element-wise
    for p, val in zip(params, result):
        try:
            python_to_param(val, p)
        except Exception:
            pass

def init(sync = 1):
    launch_func = __py_iris_callback_kernel
    dll.iris_python_init.argtypes=[__PY_IRIS_CB, ctypes.c_int32]
    dll.iris_python_init.restype = None
    dll.iris_python_init(launch_func, 0)
    status = dll.iris_init(0, None, c_int(sync))
    return status

def set_enable_profiler(flag=True):
    dll.iris_set_enable_profiler(int(flag))

def register_pin_memory(cptr, size):
    dll.iris_register_pin_memory(convert_obj_ctype(cptr)[0], convert_obj_ctype(size_t(size))[0])

def finalize():
    mem_handle_2_pobj.clear()
    return dll.iris_finalize()

def synchronize():
    return dll.iris_synchronize()

def set_nstreams(n):
    dll.call(dll.iris_set_nstreams, np.int32(n))

def set_ncopy_streams(n):
    dll.call(dll.iris_set_ncopy_streams, np.int32(n))

def nstreams():
    n_streams = dll.call_ret(dll.iris_nstreams, np.int32)
    return n_streams

def ncopy_streams():
    n_streams = dll.call_ret(dll.iris_ncopy_streams, np.int32)
    return n_streams

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
    iris_datatype_code = 0
    if type(value) == int: cvalue = byref(c_int(value))
    elif type(value) == float and size == 4: 
        cvalue = byref(c_float(value))
        iris_datatype_code = iris_float
    elif type(value) == float and size == 8: 
        cvalue = byref(c_double(value))
        iris_datatype_code = iris_double
    return dll.iris_kernel_setarg(kernel, c_int(idx), c_size_t(size) | iris_datatype_code, cvalue)

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

def convert_params(params, params_info=[], hold_params=[]):
    nparams = len(params)
    cparams = (c_void_p * nparams)()
    hold_params.clear()
    def check_size(i, s):
        if i < len(params_info) and params_info[i] != s:
            return False
        return True    
    for i in range(nparams):
        pobj = params[i]
        if type(pobj) != np.ndarray and pobj == None:
            cobj = c_void_p(0)
            cparams[i] = cobj
            hold_params.append(pobj)
        elif hasattr(params[i], 'handle') and isinstance(params[i].handle, iris_mem):
            cparams[i] = ctypes.addressof(params[i].handle)
        elif (isinstance(params[i], np.uint32) or isinstance(params[i], np.int32) or isinstance(params[i], int)) and check_size(i, 4):
            p = byref(c_int32(params[i]))
            hold_params.append(p)
            cparams[i] = cast(p, c_void_p)
        elif (isinstance(params[i], np.uint8) or isinstance(params[i], np.int8) or isinstance(params[i], int)) and check_size(i, 1):
            p = byref(c_int8(params[i]))
            hold_params.append(p)
            cparams[i] = cast(p, c_void_p)
        elif (isinstance(params[i], np.uint16) or isinstance(params[i], np.int16) or isinstance(params[i], int)) and check_size(i, 2):
            p = byref(c_short(params[i]))
            hold_params.append(p)
            cparams[i] = cast(p, c_void_p)
        elif isinstance(params[i], np.uint64) or isinstance(params[i], np.int64) or (isinstance(params[i], int)) and check_size(i, 8):
            p = byref(c_int64(params[i]))
            hold_params.append(p)
            cparams[i] = cast(p, c_void_p)
        elif (isinstance(params[i], np.float32) or isinstance(params[i], float)) and check_size(i, 4):
            p = byref(c_float(params[i]))
            cparams[i] = cast(p, c_void_p)
            hold_params.append(p)
        elif (isinstance(params[i], np.float64) or isinstance(params[i], np.double) or isinstance(params[i], float)) and check_size(i, 8):
            p = byref(c_double(params[i]))
            cparams[i] = cast(p, c_void_p)
            hold_params.append(p)
        elif type(params[i]) == size_t:
            p = byref(c_size_t(params[i]))
            hold_params.append(p)
            cparams[i] = cast(p, c_void_p)
        elif type(pobj) == np.ndarray and pobj.size > 1 and type(pobj[0]) == np.str_:
            str_list = []
            for s in str_list:
                c_str = c_char_p(s) if sys.version_info[0] == 2 else c_char_p(bytes(s, 'ascii'))
                s.append(c_str)
            s = np.array(str_list)
            cobj = s.ctypes.data_as(c_void_p)
            hold_params.append(s)
            cparams[i] = cobj
        elif type(pobj) == np.ndarray:
            cobj = pobj.ctypes.data_as(c_void_p)
            hold_params.append(pobj)
            cparams[i] = cobj
        else:
            print("error")
    return cparams
def task_kernel(task, kernel, dim, off, gws, lws, params, params_info, hold_params):
    coff = (c_size_t * dim)(*off)
    cgws = (c_size_t * dim)(*gws)
    clws = (c_size_t * dim)(*lws)
    nparams = len(params)
    cparams = convert_params(params, params_info, hold_params)

    cparams_info = (c_int * nparams)(*params_info)
    kernel_name = c_char_p(kernel) if sys.version_info[0] == 2 else c_char_p(bytes(kernel, 'ascii'))
    return dll.iris_task_kernel(task, kernel_name, c_int(dim), coff, cgws, clws, c_int(nparams), cparams, cparams_info)

def iris2py_dev(dev):
    return dev.contents.value

def iris2py(params):
    return ctypes.cast(params.contents.value, ctypes.py_object).value

def python_task_host(task, handle, func, params):
    CWRAPPER = CFUNCTYPE(c_int, POINTER(c_int64), POINTER(c_int))
    wrapped_py_func = CWRAPPER(func)
    task.params = params
    task.func = func
    task.wrapped_py_func = wrapped_py_func
    cparams = id(task.params) #(c_void_p * nparams)(*params)
    return dll.iris_task_python_host(handle, cast(wrapped_py_func, c_void_p), c_int64(cparams))

def task_host(task, func, params):
    CWRAPPER = CFUNCTYPE(c_int, c_void_p, POINTER(c_int))
    wrapped_py_func = CWRAPPER(func)
    nparams = len(params)
    cparams = (c_void_p * nparams)(*params)
    return dll.iris_task_host(task, cast(wrapped_py_func, c_void_p), cparams)

def task_h2d(task, mem, off=0, size=0, host=None):
    host_ctype = None
    if host != None:
        host_ctype = host.ctypes.data_as(c_void_p)
    return dll.iris_task_h2d(task, mem, c_size_t(off), c_size_t(size), host_ctype)

def task_params_map(task, params_map_list):
    nparams = len(params_map_list)
    c_params_map_list = (c_int * nparams)(*params_map_list)
    return dll.iris_params_map(task, c_params_map_list)

def task_d2h(task, mem, off=0, size=0, host=None):
    host_ctype = None
    if host != None:
        host_ctype = host.ctypes.data_as(c_void_p)
    return dll.iris_task_d2h(task, mem, c_size_t(off), c_size_t(size), host_ctype)

def task_d2d(task, mem, src_dev=0, off=0, size=0, host=None):
    host_ctype = None
    if host != None:
        host_ctype = host.ctypes.data_as(c_void_p)
    return dll.iris_task_d2d(task, mem, c_size_t(off), c_size_t(size), host_ctype, src_dev)

def task_h2d_full(task, mem, host):
    return dll.iris_task_h2d_full(task, mem, host.ctypes.data_as(c_void_p))

def task_d2h_full(task, mem, host):
    return dll.iris_task_d2h_full(task, mem, host.ctypes.data_as(c_void_p))

def task_submit(task, policy, sync):
    return dll.iris_task_submit(task, c_int(policy), None, c_int(sync))

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

np_obj_2_dmem = {}
mem_handle_2_pobj = {}
class dmem_null:
  def __init__(self, size=0):
    m = iris_mem()
    c_host = c_void_p(0)
    dll.iris_data_mem_create(byref(m), c_host, c_size_t(size))
    self.handle = m
    if self.handle.uid not in mem_handle_2_pobj:
        mem_handle_2_pobj[self.handle.uid] = []
    mem_handle_2_pobj[self.handle.uid].append(self)
  def release(self):
    return dll.iris_mem_release(self.handle)
  def size(self):
    dll.iris_mem_get_size.restype = c_size_t
    return dll.iris_mem_get_size(self.handle)
  def update(self, host):
    c_host = host.ctypes.data_as(c_void_p)
    dll.iris_data_mem_update(self.handle, c_host)

class dmem_base:
  track_host2dmems = {}
  def __init__(self, reuse, *args):
    self.str_data = None
    self.host = None
    self.dtype = None
    self.handle, self.type = self.dmem_create(reuse, args)
    if self.handle.uid not in mem_handle_2_pobj:
        mem_handle_2_pobj[self.handle.uid] = []
    mem_handle_2_pobj[self.handle.uid].append(self)
  def print(self, tag):
    dll.iris_data_mem_print_with_tag(self.handle, convert_obj_ctype(tag))
  def flush(self, sync=True):
      task_handle = task_create("flush")
      dll.iris_task_dmem_flush_out(task_handle, self.handle)
      task_submit(task_handle, iris_default, sync)
  def print(self):
    dll.iris_data_mem_print(self.handle)

  def host_data(self, dtype):
    ptr = dll.call_ret_ptr(dll.iris_get_dmem_valid_host, self.handle)
    buf_ptr = ctypes.cast(ptr, ctypes.POINTER(ret_type))
    np_array = np.ctypeslib.as_array(buf_ptr, shape=(length,))
    np_array = np_array.view(dtype)
    return np_array

  def dmem_create(self, reuse, *args):
    if len(args) == 1:
      c_host = None
      if len(args[0]) == 0:
        m = iris_mem()
        size=0
        c_host = c_void_p(0)
        dll.iris_data_mem_create(byref(m), c_host, c_size_t(size))
        self.handle = m
        if self.handle.uid not in mem_handle_2_pobj:
            mem_handle_2_pobj[self.handle.uid] = []
        mem_handle_2_pobj[self.handle.uid].append(self)
        return m, IRIS_DMEM
      host = args[0][0]
      host_size = 0
      dim = 0
      if isinstance(host, str):
        str_np = np.frombuffer(host.encode('utf-8') + b'\0', dtype=np.uint8)
        self.str_data = str_np.copy()
        host = self.str_data
      c_host = host.ctypes.data_as(c_void_p)
      host_size = np.array(host.shape)
      dim = len(host_size)
      size = host.nbytes
      self.dtype = host.dtype
      if reuse and c_host.value in dmem_base.track_host2dmems:
        self.host = host
        return dmem_base.track_host2dmems[c_host.value], IRIS_DMEM
      m = iris_mem()
      if dim == 1:
        chost_size = (c_size_t * dim)(*host_size)
        elem_type = get_iris_type(host.dtype)
        elem_size = host.dtype.itemsize
        dll.iris_data_mem_create_nd(byref(m), c_host, chost_size, c_int(dim), c_size_t(elem_size), c_int(elem_type))
      elif dim > 1:
        chost_size = (c_size_t * dim)(*host_size)
        elem_size = host.dtype.itemsize
        elem_type = get_iris_type(host.dtype)
        dll.iris_data_mem_create_nd(byref(m), c_host, chost_size, c_int(dim), c_size_t(elem_size), c_int(elem_type))
      self.host = host
      dmem_base.track_host2dmems[c_host.value] = m
      return m, IRIS_DMEM
    elif len(args) == 2:
      (root_mem, host) = args
      m = iris_mem()
      dll.iris_data_mem_enable_outer_dim_regions(m)
      dll.iris_data_mem_create_region(byref(m), root_mem, c_int(region))
      return m, IRIS_DMEM_REGION
    else:
      (host, off, host_size, dev_size, elem_size, dim) = args
      m = iris_mem()
      c_host = host.ctypes.data_as(c_void_p)
      coff = (c_size_t * dim)(*off)
      chost_size = (c_size_t * dim)(*host_size)
      cdev_size = (c_size_t * dim)(*dev_size)
      size = host.nbytes
      self.host = host
      dll.iris_data_mem_create_tile(byref(m), c_host, coff, chost_size, cdev_size, c_size_t(elem_size), c_int(dim))
      return m, IRIS_DMEM
  def clear(self):
    dll.iris_data_mem_clear(self.handle)
  def release(self):
    return dll.iris_mem_release(self.handle)
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
  def fetch(self, host):
    c_host = host.ctypes.data_as(c_void_p)
    dll.iris_fetch_dmem_data(self.handle, c_host)
  def host(self):
    c_host = dll.iris_get_dmem_host(self.handle)
    return c_host
  def mem(self):
    return self.mem

class dmem(dmem_base):
  def __init__(self, *args):
    super().__init__(True, *args)
  def __del__(self):
    None
  def create(np_arr):
    global np_obj_2_dmem
    if hasattr(np_arr, 'handle') and isinstance(np_arr.handle, iris_mem):
        return np_arr # Already a IRIS MEM object
    if isinstance(np_arr, np.ndarray):
        key = (np_arr.ctypes.data, np_arr.size)
        if key in np_obj_2_dmem:
            return np_obj_2_dmem[key]
        dmem_obj = dmem(np_arr)
        np_obj_2_dmem[key] = dmem_obj
        return dmem_obj
    else:
        # Do not know what it is
        return dmem(np_arr)
  def clear(self):
    dll.iris_data_mem_clear(self.handle)
  def get_region_uids(self):
    n_regions = dll.call_ret(dll.iris_data_mem_n_regions(self.handle), np.int32)
    output = []
    for i in range(n_regions):
      output.append(dll.call_ret(dll.iris_data_mem_get_region_uid, np.uint32, self.handle, np.int32(i)))
    return output

class dmem_noreuse(dmem_base):
  def __init__(self, *args):
    super().__init__(False, *args)
  def clear(self):
    dll.iris_data_mem_clear(self.handle)
  def get_region_uids(self):
    n_regions = dll.call_ret(dll.iris_data_mem_n_regions(self.handle), np.int32)
    output = []
    for i in range(n_regions):
      output.append(dll.call_ret(dll.iris_data_mem_get_region_uid, np.uint32, self.handle, np.int32(i)))
    return output

class dmem_region(dmem_base):
  def __init__(self, *args):
    super().__init__(True, args)
    self.fetch_dmem()
  def clear(self):
    dll.iris_data_mem_clear(self.handle)
  def fetch_dmem(self, handle=None):
    if handle == None:
        handle = self.handle
    dll.iris_get_dmem_for_region.restype = iris_mem
    mem_obj = dll.iris_get_dmem_for_region(handle)
    if mem_obj.uid not in mem_handle_2_pobj:
        pmem = dmem.__new__(dmem)
        pmem.type = IRIS_DMEM
        pmem.handle = mem_obj
        mem_handle_2_pobj[mem_obj.uid] = [pmem]
    pmem = mem_handle_2_pobj[mem_obj.uid][0]
    uid = pmem.uid()
    self.mem = pmem
  
class dmem_infile(dmem_base):
    def __init__(self, file_path):
        data, text_content = self.read_file(file_path)
        args = text_content
        super().__init__(True, args)
    def read_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                text_content = file.read()
                char_array = np.array(list(text_content), dtype='<U1')
                return char_array, text_content
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

class mem:
  def __init__(self, size):
    self.handle = mem_create(size)
    self.type = IRIS_MEM
    mem_handle_2_pobj[self.handle.uid] = [self]
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
    kernel_handle_2_pobj[self.handle.uid] = self
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

class command:
  def __init__(self, handle):
    self.handle = handle

  def profile(self):
    start_time = c_double()
    end_time = c_double()
    dll.iris_cmd_get_profile(self.handle, byref(start_time), byref(end_time))
    return (np.double(end_time)-np.double(start_time), np.double(start_time), np.double(end_time))

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
        if is_mem and mem_obj.uid not in mem_handle_2_pobj:
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
            mem_handle_2_pobj[mem_obj.uid] = [pmem]
            if pmem.type == IRIS_DMEM_REGION:
                pmem.fetch_dmem()
        self.mem = None
        if is_mem:
            self.mem = mem_handle_2_pobj[mem_obj.uid][0]
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
        self.params = []
        self.func = None 
        self.wrapped_py_func = None 
        self.py_func_object = None
        if len(args) == 0:
            self.handle = task_create()
            task_handle_2_pobj[self.handle.uid] = self
            return
        kernel = args[0] if len(args) > 0 else "kernel"
        dim = args[1] if len(args) > 1 else 1
        off = args[2] if len(args) > 2 else []
        gws = args[3] if len(args) > 3 else []
        lws = args[4] if len(args) > 4 else []
        params = args[5] if len(args) > 5 else []
        params_info = args[6] if len(args) > 6 else []
        if callable(kernel):
            self.handle = task_create(kernel.__name__)
        else:
            self.handle = task_create(kernel)
        self.kernel(kernel, dim, off, gws, lws, params, params_info)

    def kernel(self, kname, dim, off, gws, lws, params, params_info=[], shared_mem=0):
        if callable(kname):
            # This is python kernel function
            self.py_func_object = ctypes.py_object(kname)
            self.kernel_core(kname.__name__, dim, off, gws, lws, params, params_info, shared_mem)
            dll.iris_kernel_set_python_function(self.kernel_obj, ctypes.pointer(self.py_func_object))
            dll.iris_kernel_enable_python_interface(self.kernel_obj, iris_python_native)
        else:
            self.kernel_core(kname, dim, off, gws, lws, params, params_info, shared_mem)
        return self.last_command()

    def kernel_core(self, kernel, dim, off, gws, lws, params, params_info=[], shared_mem=0):
        if len(params_info) == 0  and len(params) != len(params_info):
            return self.kernel_core2(kernel, dim, off, gws, lws, params, shared_mem)
        else:
            return self.kernel_core1(kernel, dim, off, gws, lws, params, params_info, shared_mem)

    def kernel_core1(self, kernel, dim, off, gws, lws, params, params_info=[], shared_mem=0):
        task_kernel(self.handle, kernel, dim, off, gws, lws, params, params_info, self.params)
        self.kernel_obj = dll.call_ret(dll.iris_task_get_last_cmd_kernel, iris_kernel, self.handle)
        if shared_mem !=0:
            dll.iris_kernel_set_shared_memory_size(self.kernel_obj, c_size_t(shared_mem))
        return self.kernel_obj

    def kernel_core2(self, kernel, dim, off, gws, lws, params_list, shared_mem=0):
        coff = (c_size_t * dim)(*off)
        cgws = (c_size_t * dim)(*gws)
        clws = (c_size_t * dim)(*lws)
        nparams = len(params_list)
        self.params = []
        cparams = (c_void_p * nparams)()
        task_handle_2_pobj[self.handle.uid] = self
        params_info = []
        flush_objs = []
        for i, pobj in enumerate(params_list):
            if type(pobj) == tuple:
                if hasattr(pobj[0], 'handle') and isinstance(pobj[0].handle, iris_mem):
                    self.params.append(pobj[0])
                    cparams[i] = ctypes.addressof(pobj[0].handle)
                    if len(pobj) == 2 and pobj[1] == iris_flush:
                        params_info.append(iris_w)
                    else:
                        params_info.append(pobj[1])
                    if len(pobj) == 3 and pobj[2] == iris_flush:
                        flush_objs.append(pobj[0].handle)
                    if len(pobj) == 2 and pobj[1] == iris_flush:
                        flush_objs.append(pobj[0].handle)
                else:
                    # (NUMPY Object, modes of use, <optional flush>)
                    # Should be numpy array
                    host = pobj[0]
                    dobj = dmem.create(host)
                    self.params.append(dobj)
                    cparams[i] = ctypes.addressof(dobj.handle)
                    if len(pobj) == 2 and pobj[1] == iris_flush:
                        params_info.append(iris_w)
                    else:
                        params_info.append(pobj[1])
                    if len(pobj) == 2 and pobj[1] == iris_flush:
                        flush_objs.append(dobj.handle)
                    if len(pobj) == 3 and pobj[2] == iris_flush:
                        flush_objs.append(dobj.handle)
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
        self.kernel_obj = dll.iris_task_kernel(self.handle, kernel_name, c_int(dim), coff, cgws, clws, c_int(nparams), cparams, cparams_info)
        self.kernel_obj = dll.call_ret(dll.iris_task_get_last_cmd_kernel, iris_kernel, self.handle)
        if shared_mem !=0:
            dll.iris_kernel_set_shared_memory_size(self.kernel_obj, c_size_t(shared_mem))
        for pobj in flush_objs:
            dll.iris_task_dmem_flush_out(self.handle, pobj)
        return self.kernel_obj

    def flush(self, mem):
        dll.iris_task_dmem_flush_out(self.handle, mem.handle)
        return self.last_command()

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
        dll.iris_task_set_device(self.handle, convert_obj_ctype(device)[0])
    def set_stream(self, stream): # Set stream affinity
        dll.iris_task_set_stream_affinity(self.handle, convert_obj_ctype(stream)[0])
    def set_policy(self, policy):
        dll.iris_task_set_policy(self.handle, convert_obj_ctype(policy)[0])
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
        if ckernel.uid not in kernel_handle_2_pobj:
            pkernel = kernel.__new__(kernel)
            pkernel.handle = ckernel
            kernel_handle_2_pobj[ckernel.uid] = pkernel
        return kernel_handle_2_pobj[ckernel.uid]

    def get_ndepends(self):
        dll.iris_task_get_dependency_count.restype = c_int
        ntasks = dll.iris_task_get_dependency_count(self.handle)
        return ntasks

    def last_command(self):
        dll.iris_task_get_last_cmd.restype = iris_kernel
        cmd = dll.iris_task_get_last_cmd(self.handle)
        return command(cmd)

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
    def wait(self):
        task_wait(self.handle)
    def params_map(self, params_map_list):
        task_params_map(self.handle, params_map_list)
    def dmem2dmem(self, src_mem, dst_mem):
        src_d_mem = dmem.create(src_mem)
        dst_d_mem = dmem.create(dst_mem)
        dll.iris_task_dmem2dmem(self.handle, src_d_mem.handle, dst_d_mem.handle)
        return self.last_command()
    def mem_in(self, mem, off=0, size=0, host=None):
        d_mem = dmem.create(mem)
        task_h2d(self.handle, d_mem.handle, off, size, host)
        return self.last_command()
    def h2d(self, mem, off=0, size=0, host=None):
        d_mem = dmem.create(mem)
        task_h2d(self.handle, d_mem.handle, off, size, host)
        return self.last_command()
    def mem_out(self, mem, off=0, size=0, host=None):
        d_mem = dmem.create(mem)
        task_d2h(self.handle, d_mem.handle, off, size, host)
        return self.last_command()
    def d2h(self, mem, off=0, size=0, host=None):
        d_mem = dmem.create(mem)
        task_d2h(self.handle, d_mem.handle, off, size, host)
        return self.last_command()
    def d2d(self, mem, src_dev=0, off=0, size=0, host=None):
        d_mem = dmem.create(mem)
        task_d2d(self.handle, d_mem.handle, src_dev, off, size, host)
        return self.last_command()
    def h2d_full(self, mem, host):
        d_mem = dmem.create(mem)
        task_h2d_full(self.handle, d_mem.handle, host)
        return self.last_command()
    def d2h_full(self, mem, host):
        d_mem = dmem.create(mem)
        task_d2h_full(self.handle, d_mem.handle, host)
        return self.last_command()
    def pyhost(self, func, params):
        python_task_host(self, self.handle, func, params)
        return self.last_command()
    def host(self, func, params):
        task_host(self.handle, func, params)
        return self.last_command()
    def submit(self, policy=iris_default, sync = 1):
        task_submit(self.handle, policy, sync)
    def task_name(self):
        dll.iris_task_get_name(self.handle)
    def kernel_name(self):
        dll.iris_task_get_kernel_name(self.handle)
    def kernel_params(self):
        dll.iris_task_get_kernel_params(self.handle)
    def release(self):
        dll.iris_task_release(self.handle)
    def retain(self):
        dll.iris_task_retain(self.handle, c_int(True))
    def inputs(self):
        dll.iris_task_get_inputs(self.handle)
    def outputs(self):
        dll.iris_task_get_inputs(self.handle)

def convert_ctasks(c_tasks):
    ptasks = []
    for c_task in c_tasks:
        if c_task.uid not in task_handle_2_pobj:
            p_task = task.__new__(task)
            p_task.params = []
            p_task.handle = c_task
            task_handle_2_pobj[c_task.uid] = p_task
        ptasks.append(task_handle_2_pobj[c_task.uid])
    return ptasks
class graph:
    def __init__(self, tasks=[], device=iris_default, opt=None):
        self.handle = graph_create()
        for each_task in tasks:
            self.add_task(each_task, device, opt)
    def load(self, json_file='graph.json', params=[]):
        c_json_file = c_char_p(json_file) if sys.version_info[0] == 2 else c_char_p(bytes(json_file, 'ascii'))
        self.handle = iris_graph()
        cparams = convert_params(params, params_info=[], hold_params=[])
        d = dll.iris_graph_create_json(c_json_file, cparams, byref(self.handle))
        return d
    def retain(self):
        dll.iris_graph_retain(self.handle, c_int(True))
    def release(self):
        dll.iris_graph_release(self.handle)
    def free(self):
        dll.iris_graph_free(self.handle)
    def wait(self):
        dll.iris_graph_wait(self.handle)
    def submit(self, policy=iris_default, sync=1, order=None):
        if type(order) == np.ndarray or order != None:
            order_np = order
            if type(order) != np.ndarray:
                order_np = np.int32(order)
            conv_order_np = convert_obj_ctype(order_np)
            dll.iris_graph_submit_with_order(self.handle, conv_order_np[0], c_int(policy), c_int(sync))
        else:
            dll.iris_graph_submit(self.handle, c_int(policy), c_int(sync))
    def submit_time(self, policy=iris_default, sync=1, order=None):
        #stime = np.double(0.0)
        #cobj = byref(c_double(stime))
        stime = np.zeros((1), np.double)
        cobj = stime.ctypes.data_as(c_void_p)
        if type(order) == np.ndarray or order != None:
            order_np = order
            if type(order) != np.ndarray:
                order_np = np.int32(order)
            conv_order_np = convert_obj_ctype(order_np)
            dll.iris_graph_submit_with_order_and_time(self.handle, conv_order_np[0], cobj, c_int(policy), c_int(sync))
        else:
            dll.iris_graph_submit_with_time(self.handle, cobj, c_int(policy), c_int(sync))
        return stime[0]
    def add_task(self, task, policy=iris_default, opt=None):
        c_opt = c_void_p(0)
        if opt != None:
           c_opt = c_char_p(opt) if sys.version_info[0] == 2 else c_char_p(bytes(opt, 'ascii'))
        dll.iris_graph_task(self.handle, task.handle, c_int(policy), c_opt)

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
        dll.call_ret(dll.iris_get_graph_dependency_adj_matrix, np.int32, self.handle, dep_graph)
        #print("Dependency matrix", dep_graph)
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
        dll.call_ret(dll.iris_get_graph_2d_comm_adj_matrix, np.int32, self.handle, comm_2d)
        print("Extracting 2d communication cost")
        #print("Communication cost matrix", comm_2d)
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
            dll.call_ret(dll.iris_calibrate_compute_cost_adj_matrix_only_for_types, np.int32, self.handle, comp_2d)
        else:
            dll.call_ret(dll.iris_calibrate_compute_cost_adj_matrix, np.int32, self.handle, comp_2d)
        print("Completed computation cost matrix calibration", comp_2d)
        if pdf:
            task_names = self.get_task_names()
            task_uids =  self.get_task_uids()
            import pandas as pd
            df = pd.DataFrame(comp_2d, columns=col_names)
            df['task_name'] = task_names
            df['task_uid'] = task_uids
            return df
        return comp_2d
class json_graph(graph):
    def __init__(self, json_file='graph.json', params=[]):
        self.load(json_file, params)

