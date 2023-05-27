#
# Author : Narasinga Rao Miniskar
# Date   : Jun 16 2022
# Contact for bug : miniskarnr@ornl.gov
#
import re
import argparse
import pdb
import regex
import os
import tempfile

IRIS_CREATE_APIS = 3
IRIS_TASK_APIS_CPP = 2
IRIS_TASK_APIS = 1
IRIS_TASK_NO_SUBMIT = 0
IRIS_TASK = 0
IRIS_TASK_DEFAULT = 0

C_CORE_API=0
C_TASK_API=1
CPP_CORE_API=2
CPP_TASK_API=3
global_arguments_start_index=7
valid_params = { 'PARAM' : 1, 'PARAM_CONST' : 1, 'IN_TASK' : 1, 'IN_TASK_OFFSET' : 1, 'OUT_TASK_OFFSET' : 1, 'OUT_TASK' : 1, 'IN_OUT_TASK' : 1, 'IN_OUT_TASK_OFFSET' : 1, 'VEC_PARAM' : 1, 'VS_CODE_GEN' : 1, 'TILED_CODE_GEN' : 1, 'CPP_TASK_TEMPLATE' : 1 }

global_res= { 
    "iris_all" : 0xFFFFFFFF, 
    "iris_cpu"    : 0x01, 
    "iris_dsp"    : 0x02, 
    "iris_fpga"   : 0x04, 
    "iris_gpu"    : 0x08,
    "iris_cuda"   : 0x10,
    "iris_nvidia" : 0x10,
    "iris_hip"    : 0x20,
    "iris_amd"    : 0x20,
    "iris_opencl" : 0x40,
}

def remove_spaces(api_name):
    api_name = re.sub(r'"', '', api_name)
    api_name = re.sub(r'\s*', '', api_name)
    return api_name

def WriteFile(output_file, lines, tag=''):
    print("Generating "+tag+" file: "+output_file)
    wfh = open(output_file, 'w')
    for l in lines:
        wfh.write(l+"\n")
    wfh.close()

def getPyExprString(l):
    if type(l) == list:
        p = []
        for i in l:
            if type(i) == list:
                p.append(" ( ")
                p.append(getPyExprString(i))
                p.append(" ) ")
            else:
                p.append(i)
        return " ".join(p)
    else:
        return l

def extractTaskCallsCore(filename, args):
    f = tempfile.NamedTemporaryFile(delete=False)
    print(f"[Cmd] g++ -x c++ -E {filename} -Uiris_cpu -Uiris_dsp -o {f.name} {args.compile_options} -DUNDEF_IRIS_MACROS -DEXTRACT_MACROS -std=c++11")
    os.system(f"g++ -x c++ -E {filename} -Uiris_cpu -Uiris_dsp -o {f.name} {args.compile_options} -DUNDEF_IRIS_MACROS -DEXTRACT_MACROS -std=c++11")
    if not os.path.exists(f.name):
        return ""
    fh = open(f.name, "r")
    lines = fh.readlines()
    single_line = " ".join(lines);
    single_line = re.sub(r'\r', '', single_line)
    single_line = re.sub(r'\n', '', single_line)
    result0 = regex.findall(r'''IRIS_SINGLE_TASK
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    result1 = regex.findall(r'''IRIS_TASK_NO_DT
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    result2 = regex.findall(r'''IRIS_TASK
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    result3 = regex.findall(r'''IRIS_TASK_NO_SUBMIT
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    result4 = regex.findall(r'''IRIS_TASK_APIS
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    result5 = regex.findall(r'''IRIS_TASK_APIS_CPP
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    result6 = regex.findall(r'''IRIS_CREATE_APIS
        (?<rec> #capturing group rec
        \( #open parenthesis
        (?: #non-capturing group
        [^()]++ #anyting but parenthesis one or more times without backtracking
        | #or
        (?&rec) #recursive substitute of group rec
        )*
        \) #close parenthesis
        )
        ''',single_line,flags=regex.VERBOSE)
    def add_code(result, code=IRIS_TASK_DEFAULT):
        result_out = [ '('+str(code)+', '+r[1:] for r in result ]
        return result_out
    fh.close()
    os.unlink(f.name)
    return add_code(result0)+\
           add_code(result1)+\
           add_code(result2)+\
           add_code(result3)+\
           add_code(result4, IRIS_TASK_APIS)+\
           add_code(result5, IRIS_TASK_APIS_CPP)+\
           add_code(result6, IRIS_CREATE_APIS)

def extractTaskCalls(args):
    output  = []
    for in_file in args.input:
        output = output + extractTaskCallsCore(in_file, args)
    return output

def parseFnParams(text):
    fn_name, nstack = parseFnParamsCore(text)
    arguments_start_index = find_start_index(nstack)
    i = arguments_start_index
    nnstack = nstack[0:i]
    while (i<len(nstack)):
        lfn_name, lnstack = parseFnParamsCore(nstack[i])
        nnstack.append(lfn_name)
        nnstack.append(lnstack)
        i = i+1
    def format_list(data):
        for index, n in enumerate(data):
            if type(n) == list:
                format_list(n)
            else:
                data[index] = re.sub(r'^\s*', '', data[index])
                data[index] = re.sub(r'\s*$', '', data[index])
    format_list(nnstack)
    return nnstack

def parseFnParamsCore(text):
    left = re.compile(r'\(')
    right = re.compile(r'\)')
    sep = re.compile(r',')
    level = 0
    fn_name = ''    
    stack = []
    for y in text:
        if level == 0 and not left.match(y):
            fn_name += y
        if left.match(y):
            level=level+1
            if level > 1:
                stack[-1] += y
        elif right.match(y):
            if level > 1:
                stack[-1] += y
            level=level-1
        elif sep.match(y) and level == 1:
            stack.append("")
        elif level > 0:
            if len(stack) == 0:
                stack.append("")
            stack[-1] += y
    stack = [ x for x in stack if x != '']
    return fn_name, stack
#'(task0, "saxpy", target_dev, SIZE,           OUT_TASK(Z, int32_t *, sizeof(int32_t)*SIZE),           IN_TASK(X, int32_t *, sizeof(int32_t)*SIZE),           IN_TASK(Y, int32_t *, sizeof(i
#nt32_t)*SIZE),           PARAM(A, int32_t),           PARAM(SIZE, int, iris_cpu),           PARAM(dspUsecPtr, int32_t*, (iris_cpu || iris_dsp)),           PARAM(dspCycPtr, int32_t*, brisb
#ane_cpu || iris_dsp))\n'
#["task0", "saxpy", "target_dev", "SIZE", { "fn" : "OUT_TASK", "params": [ "Z", "int32_t *", { "fn" : "*", "params": { { "fn" : "sizeof", "params": "int32_t" }, "SIZE" }}, 
def preprocess_data(data):
    #print(f"Input:{data}")
    if type(data) == list:
        data = [ preprocess_data(x) for x in data ]
    elif isinstance(data, dict):
        dh = {}
        for k,v in data.items():
            dh[preprocess_data(k)] = preprocess_data(v)
        data = dh
    else:
        data = re.sub(r'^\s*', '', data)
    return data

def read_metadata_file(args, inputs):
    codes = []
    for file in inputs:
        print("Reading input: "+file)
        fh = open(file, 'r')
        lcodes = fh.readlines()
        for code in lcodes:
            codes.append(code)
        fh.close()
    return codes

def generateIrisInterfaceCode(args, codes):
    #print("\n".join(codes))
    data = []
    data_hash = {}
    header_data_hash = {}
    for code_msg in codes:
        d_code = [parseFnParams(code_msg)]
        code = int(d_code[0][0])
        d = [d_code[0][1:]]
        if code == IRIS_TASK_DEFAULT:
            kname = d[0][1]
            kname = re.sub(r'"', '', kname)
            kname = re.sub(r'\s*', '', kname)
            d[0][1] = kname
            data.append(d[0])
            data_hash[kname] = preprocess_data(d[0])
        elif code == IRIS_TASK_APIS:
            (core_api_name, task_api_name) = d[0][0], d[0][1]
            core_api_name = remove_spaces(core_api_name)
            task_api_name = remove_spaces(task_api_name)
            d = [['task0'] + d[0][2:]]
            kname_original = d[0][1]
            kname = remove_spaces(d[0][1])
            d[0][1] = kname
            data.append(d[0])
            data_hash[kname] = preprocess_data(d[0])
            data_details_hash = extract_header_data_details_hash(args, data_hash[kname])
            header_data_hash[core_api_name] = (C_CORE_API, data_hash[kname], data_details_hash, code, kname_original, (core_api_name, task_api_name))
            header_data_hash[task_api_name] = (C_TASK_API, data_hash[kname], data_details_hash, code, kname_original, (core_api_name, task_api_name))
        elif code == IRIS_TASK_APIS_CPP or code == IRIS_CREATE_APIS:
            (cpp_api_name, core_api_name, task_api_name) = d[0][0], d[0][1], d[0][2]
            cpp_api_name  = remove_spaces(cpp_api_name)
            core_api_name = remove_spaces(core_api_name)
            task_api_name = remove_spaces(task_api_name)
            d = [['task0'] + d[0][3:]]
            kname_original = d[0][1]
            kname = remove_spaces(d[0][1])
            d[0][1] = kname
            data.append(d[0])
            data_hash[kname] = preprocess_data(d[0])
            data_details_hash = extract_header_data_details_hash(args, data_hash[kname])
            header_data_hash[core_api_name] = (C_CORE_API, data_hash[kname], data_details_hash, code, kname_original, (core_api_name, task_api_name))
            header_data_hash[task_api_name] = (C_TASK_API, data_hash[kname], data_details_hash, code, kname_original, (core_api_name, task_api_name))
            header_data_hash[(cpp_api_name, core_api_name)] =  (CPP_CORE_API, data_hash[kname], data_details_hash, code, kname_original, (core_api_name, task_api_name))
            header_data_hash[(cpp_api_name, task_api_name)] =  (CPP_TASK_API, data_hash[kname], data_details_hash, code, kname_original, (core_api_name, task_api_name))
    k_hash = {}
    k_index = 0
    for k,v in data_hash.items():
        k_hash[k] = k_index
        k_index = k_index+1
    lines = []
    lines.append('''
#ifndef __IRIS_CPU_DSP_INTERFACE_H__
#define __IRIS_CPU_DSP_INTERFACE_H__

#include <stdio.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
    ''')
    if args.thread_safe != 1:
        lines.append('''
static int iris_kernel_idx = -1;

        ''')
    lines.append('''
#ifdef ENABLE_IRIS_HEXAGON_APIS
#define iris_kernel_lock   iris_hexagon_lock
#define iris_kernel_unlock iris_hexagon_unlock
#define iris_kernel iris_hexagon_kernel
#define iris_setarg iris_hexagon_setarg
#define iris_setmem iris_hexagon_setmem
#define iris_launch iris_hexagon_launch
#define iris_kernel_with_obj iris_hexagon_kernel_with_obj
#define iris_setarg_with_obj iris_hexagon_setarg_with_obj 
#define iris_setmem_with_obj iris_hexagon_setmem_with_obj 
#define iris_launch_with_obj iris_hexagon_launch_with_obj
            ''')
    for k,v in data_hash.items():
        lines.append("#define iris_kernel_"+k+" irishxg_"+k)
    lines.append('''
#define HANDLE irishxg_handle_stub(), 
#define HANDLE_WITH_OBJ irishxg_handle_stub(), 
#define HANDLETYPE uint64_t,
#define DEVICE_NUM_TYPE 
#define DEVICE_NUM
#define StreamType
#define STREAM_VAR
#endif //IRIS_HEXAGON
            ''')
    lines.append('''
#ifndef ENABLE_IRIS_HEXAGON_APIS
#ifndef ENABLE_IRIS_HOST2HIP_APIS
#ifndef ENABLE_IRIS_HOST2CUDA_APIS
#ifndef ENABLE_IRIS_HOST2OPENCL_APIS
#define ENABLE_IRIS_OPENMP_APIS
#endif
#endif
#endif
#endif
            ''')
    lines.append('''
#ifdef ENABLE_IRIS_OPENMP_APIS 
#include "iris/iris_openmp.h"
#define iris_kernel_lock   iris_openmp_lock
#define iris_kernel_unlock iris_openmp_unlock
#define iris_kernel iris_openmp_kernel
#define iris_setarg iris_openmp_setarg
#define iris_setmem iris_openmp_setmem
#define iris_launch iris_openmp_launch
#define iris_kernel_with_obj iris_openmp_kernel_with_obj
#define iris_setarg_with_obj iris_openmp_setarg_with_obj 
#define iris_setmem_with_obj iris_openmp_setmem_with_obj 
#define iris_launch_with_obj iris_openmp_launch_with_obj
#define HANDLE
#define HANDLE_WITH_OBJ
#define HANDLETYPE 
#define DEVICE_NUM
#define DEVICE_NUM_TYPE 
#define STREAM_VAR
            ''')
    for k,v in data_hash.items():
        lines.append("#define iris_kernel_"+k+" "+k)
    lines.append('''
#endif //ENABLE_IRIS_OPENMP_APIS
            ''')
    lines.append('''
#ifdef ENABLE_IRIS_HOST2OPENCL_APIS 
#include "iris/iris_host2opencl.h"
#define iris_kernel_lock   iris_host2opencl_lock
#define iris_kernel_unlock iris_host2opencl_unlock
#define iris_kernel iris_host2opencl_kernel
#define iris_setarg iris_host2opencl_setarg
#define iris_setmem iris_host2opencl_setmem
#define iris_launch iris_host2opencl_launch
#define iris_kernel_with_obj iris_host2opencl_kernel_with_obj
#define iris_setarg_with_obj iris_host2opencl_setarg_with_obj 
#define iris_setmem_with_obj iris_host2opencl_setmem_with_obj 
#define iris_launch_with_obj iris_host2opencl_launch_with_obj
#define iris_kernel_set_queue_with_obj iris_host2opencl_set_queue_with_obj
#define iris_kernel_get_queue_with_obj iris_host2opencl_get_queue_with_obj
#define HANDLE  iris_host2opencl_get_queue(), 
#define HANDLE_WITH_OBJ iris_host2opencl_get_queue_with_obj(obj), 
#define HANDLETYPE void *,
#define DEVICE_NUM devno,
#define DEVICE_NUM_TYPE int,
#define STREAM_VAR
            ''')
    lines.append("#ifdef HOST2OPENCL")
    for k,v in data_hash.items():
        lines.append("#define "+k+" HOST2OPENCL("+k+")")
    lines.append("#endif //HOST2OPENCL")
    for k,v in data_hash.items():
        lines.append("#define iris_kernel_"+k+" "+k)
    lines.append('''
#endif //ENABLE_IRIS_HOST2OPENCL_APIS
            ''')
    lines.append('''
#ifdef ENABLE_IRIS_HOST2HIP_APIS 
#include "iris/iris_host2hip.h"
#define iris_kernel_lock   iris_host2hip_lock
#define iris_kernel_unlock iris_host2hip_unlock
#define iris_kernel iris_host2hip_kernel
#define iris_setarg iris_host2hip_setarg
#define iris_setmem iris_host2hip_setmem
#define iris_launch iris_host2hip_launch
#define iris_kernel_with_obj iris_host2hip_kernel_with_obj
#define iris_setarg_with_obj iris_host2hip_setarg_with_obj 
#define iris_setmem_with_obj iris_host2hip_setmem_with_obj 
#define iris_launch_with_obj iris_host2hip_launch_with_obj
#define HANDLE 
#define HANDLE_WITH_OBJ
#define HANDLETYPE 
#define DEVICE_NUM devno,
#define DEVICE_NUM_TYPE int,
#ifndef IRIS_ENABLE_STREAMS
#define STREAM_VAR
#else
#define StreamType hipStream_t
#define STREAM_VAR stream,
#endif
            ''')
    lines.append("#ifdef HOST2HIP")
    for k,v in data_hash.items():
        lines.append("#define "+k+" HOST2HIP("+k+")")
    lines.append("#endif //HOST2HIP")
    for k,v in data_hash.items():
        lines.append("#define iris_kernel_"+k+" "+k)
    lines.append('''
#endif //ENABLE_IRIS_HOST2HIP_APIS
            ''')
    lines.append('''
#ifdef ENABLE_IRIS_HOST2CUDA_APIS 
#include "iris/iris_host2cuda.h"
#define iris_kernel_lock   iris_host2cuda_lock
#define iris_kernel_unlock iris_host2cuda_unlock
#define iris_kernel iris_host2cuda_kernel
#define iris_setarg iris_host2cuda_setarg
#define iris_setmem iris_host2cuda_setmem
#define iris_launch iris_host2cuda_launch
#define iris_kernel_with_obj iris_host2cuda_kernel_with_obj
#define iris_setarg_with_obj iris_host2cuda_setarg_with_obj 
#define iris_setmem_with_obj iris_host2cuda_setmem_with_obj 
#define iris_launch_with_obj iris_host2cuda_launch_with_obj
#define HANDLE 
#define HANDLE_WITH_OBJ
#define HANDLETYPE 
#define DEVICE_NUM devno,
#define DEVICE_NUM_TYPE int,
#ifndef IRIS_ENABLE_STREAMS
#define STREAM_VAR
#else
#define StreamType CUstream
#define STREAM_VAR stream,
#endif
            ''')
    lines.append("#ifdef HOST2CUDA")
    for k,v in data_hash.items():
        lines.append("#define "+k+" HOST2CUDA("+k+")")
    lines.append("#endif //HOST2CUDA")
    for k,v in data_hash.items():
        lines.append("#define iris_kernel_"+k+" "+k)
    lines.append('''
#endif //ENABLE_IRIS_HOST2CUDA_APIS
            ''')
    appendKernelSignature(args, lines, data_hash, k_hash)
    sig_lines = []
    cu_type_lines = {}
    appendKernelSignatureHeaderFile(args, sig_lines, header_data_hash)
    generateTaskAPIs(args, cu_type_lines, header_data_hash)
    #lines = lines + sig_lines
    k_sig_lines = [ ]
    k_sig_lines += [ """
/**
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */
    """]
    k_sig_lines += [ "#pragma once"]
    k_sig_lines += sig_lines
    if os.path.exists('codegen_task_template.h'):
        k_sig_lines += [ "#include <codegen_task_template.h>"]
    if os.path.exists('codegen_tiled.h'):
        k_sig_lines += [ "#include <codegen_tiled.h>"]
    WriteFile(args.signature_file, k_sig_lines, "Consolidated CPU/DSP interface code for all kernels in header")
    appendStructure(args, lines, data_hash, k_hash)
    if args.thread_safe == 1:
        appendSetKernelWrapperFunctions(args, lines, data_hash, k_hash)
        appendHandleWrapperFunctions(args, lines, data_hash, k_hash)
    appendKernelNamesFunction(args, lines, data_hash, k_hash)
    for k,v in data_hash.items():
        appendKernelSetArgMemParamFunctions(args, lines, k, v)
    appendKernelSetArgMemGlobalFunctions(args, lines, data_hash, k_hash)
    appendKernelLaunchFunction(args, lines, data_hash, k_hash)
    lines.append('''
#ifdef __cplusplus
} /* end of extern "C" */
#endif
#endif //__IRIS_CPU_DSP_INTERFACE_H__
            ''')
    for k,v in data_hash.items():
        print("Generated CPU/DSP interface code for kernel:"+k)
    return lines

def getKernelParamVairable(k):
    return "__k_"+k+"_args"

def is_expression_satisfied(expr, constraint='iris_all'):
    global global_res
    res = global_res
    if not re.match(r'.*iris_', expr):
        return False
    expr = re.sub('\|\|', ' | ', expr)
    expr = re.sub('\&\&', ' & ', expr)
    expr = expr + " & " + constraint
    #print("EXPR: "+expr)
    expr_eval = eval(expr, res.copy())
    if expr_eval == res['iris_all']:
        return True
    cu_op = {}
    for k,v in res.items():
        flag = expr_eval & res[k]
        if k == 'iris_gpu':
            cu_op['iris_nvidia'] = 1
            cu_op['iris_amd']    = 1
            cu_op['iris_fpga']   = 1
        else:
            cu_op[k] = 1
    for k,v in cu_op.items():
        flag = expr_eval & res[k]
        if not flag:
            continue
        if k == constraint:
            return True
    return False
def appendConditionalParameters(expr, fdt, params, lines, constraint='iris_all', macros_flag=True):
    global global_res
    res = global_res
    if not re.match(r'.*iris_', expr):
        params.append(fdt)
        return
    expr = re.sub('\|\|', ' | ', expr)
    expr = re.sub('\&\&', ' & ', expr)
    #print("EXPR: "+expr)
    expr_eval = eval(expr, res.copy())
    if expr_eval == res['iris_all']:
        params.append(fdt)
        return
    if len(params) > 0:
         lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
    params.clear()
    cu_op = {}
    for k,v in res.items():
        flag = expr_eval & res[k]
        if k == 'iris_gpu':
            cu_op['iris_nvidia'] = 1
            cu_op['iris_amd']    = 1
            cu_op['iris_fpga']   = 1
        else:
            cu_op[k] = 1
    for k,v in cu_op.items():
        flag = expr_eval & res[k] & global_res[constraint]
        if not flag:
            continue
        if k == 'iris_dsp':
            if macros_flag:
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            if macros_flag:
                lines.append("#endif") 
        elif k == 'iris_cpu':
            if macros_flag:
                lines.append("#ifdef ENABLE_IRIS_OPENMP_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            if macros_flag:
                lines.append("#endif")
        elif k == 'iris_nvidia':
            if macros_flag:
                lines.append("#ifdef ENABLE_IRIS_HOST2OPENCL_APIS")
                lines.append("\t\t\t\t"+fdt+",")
                lines.append("#endif")
                lines.append("#ifdef ENABLE_IRIS_HOST2CUDA_APIS")
                lines.append("\t\t\t\t"+fdt+",")
                lines.append("#endif")
            else:
                lines.append("\t\t\t\t"+fdt+",")
        elif k == 'iris_amd':
            if macros_flag:
                lines.append("#ifdef ENABLE_IRIS_HOST2OPENCL_APIS")
                lines.append("\t\t\t\t"+fdt+",")
                lines.append("#endif")
                lines.append("#ifdef ENABLE_IRIS_HOST2HIP_APIS")
                lines.append("\t\t\t\t"+fdt+",")
                lines.append("#endif")
            else:
                lines.append("\t\t\t\t"+fdt+",")

def get_parameter_types(v, lines=[], constraint='iris_all'):
    global valid_params, global_res
    arguments = []
    arguments_variable_hash = {}
    arguments_start_index = find_start_index(v)
    i = arguments_start_index
    one_param_exists = False
    expression_index = { "PARAM" : 2, "PARAM_CONST" : 2, "VEC_PARAM" : 2, 
        "IN_TASK" : -1, "IN_TASK_OFFSET" : -1, 
        "OUT_TASK" : -1, "OUT_TASK_OFFSET" : -1, 
        "IN_OUT_TASK" : -1, "IN_OUT_TASK_OFFSET" : -1 }
    elem_dt_type_index = { "PARAM" : 1, "PARAM_CONST" : 1, "VEC_PARAM" : 1, 
        "IN_TASK" : 2, "IN_TASK_OFFSET" : 2, 
        "OUT_TASK" : 2, "OUT_TASK_OFFSET" : 2, 
        "IN_OUT_TASK" : 2, "IN_OUT_TASK_OFFSET" : 2 }
    dt_type_index = { "PARAM" : 1, "PARAM_CONST" : 1, "VEC_PARAM" : 1, 
        "IN_TASK" : 1, "IN_TASK_OFFSET" : 2, 
        "OUT_TASK" : 2, "OUT_TASK_OFFSET" : 2, 
        "IN_OUT_TASK" : 2, "IN_OUT_TASK_OFFSET" : 2 }
    while (i<len(v)):
        f_param = remove_spaces(v[i])
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = v[i]
        if f_param == 'VS_CODE_GEN' or \
           f_param == 'TILED_CODE_GEN' or \
           f_param == 'CPP_TASK_TEMPLATE':
            i = i+1
            continue
        expr = 'iris_all' 
        if expression_index[f_param] == -1 or len(f_details) > expression_index[f_param]:
            expr = getPyExprString(f_details[expression_index[f_param]])
            if expr not in global_res:
                expr = 'iris_all'
        if is_expression_satisfied(expr, constraint):
            elem_dt_index = elem_dt_type_index[f_param]
            dt_index = 1
            arguments.append((f_details[0], f_param, f_details[dt_index], f_details[elem_dt_index], expr, f_details))
            arguments_variable_hash[f_details[0]] = (f_param, f_details[dt_index], f_details[elem_dt_index], expr, f_details)
        i = i+1
    return arguments, arguments_variable_hash
def add_conditional_parameters_to_kernel(kvar, k, v, lines=[]):
    global valid_params, global_res
    def InsertConditionalParameters(kvar, k, expr, f_details, params, lines):
        appendConditionalParameters(expr, kvar+f_details[0], params, lines)

    params = []
    arguments_start_index = find_start_index(v)
    i = arguments_start_index
    one_param_exists = False
    while (i<len(v)):
        f_param = remove_spaces(v[i])
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = v[i]
        if f_param == 'PARAM' and len(f_details)>2:
            expr = getPyExprString(f_details[2])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            one_param_exists = True
        elif f_param == 'PARAM_CONST' and len(f_details)>3:
            expr = getPyExprString(f_details[2])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            one_param_exists = True
        elif f_param == 'VEC_PARAM' and len(f_details)>2:
            expr = getPyExprString(f_details[2])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            one_param_exists = True
        elif f_param == 'IN_TASK':
            expr = getPyExprString(f_details[-1])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            if len(params) > 0:
                 lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t("+kvar+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
            lines.append("#endif")
            params.clear()
            one_param_exists = True
        elif f_param == 'IN_TASK_OFFSET':
            expr = getPyExprString(f_details[-1])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            if len(params) > 0:
                 lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t("+kvar+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
            lines.append("#endif")
            params.clear()
            one_param_exists = True
        elif f_param == 'OUT_TASK':
            expr = getPyExprString(f_details[-1])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            if len(params) > 0:
                 lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t("+kvar+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
            lines.append("#endif")
            params.clear()
            one_param_exists = True
        elif f_param == 'OUT_TASK_OFFSET':
            expr = getPyExprString(f_details[-1])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            if len(params) > 0:
                 lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t("+kvar+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
            lines.append("#endif")
            params.clear()
            one_param_exists = True
        elif f_param == 'IN_OUT_TASK':
            expr = getPyExprString(f_details[-1])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            if len(params) > 0:
                 lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t("+kvar+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
            lines.append("#endif")
            params.clear()
            one_param_exists = True
        elif f_param == 'IN_OUT_TASK_OFFSET':
            expr = getPyExprString(f_details[-1])
            InsertConditionalParameters(kvar, k, expr, f_details, params, lines)
            if len(params) > 0:
                 lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t("+kvar+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
            lines.append("#endif")
            params.clear()
            one_param_exists = True
        elif f_param == 'VS_CODE_GEN':
            None
        elif f_param == 'TILED_CODE_GEN':
            None
        elif f_param == 'CPP_TASK_TEMPLATE':
            None
        else:
            params.append(kvar+f_details[0])
            one_param_exists = True
        i = i+1
    if len(params) > 0:
        lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
    return lines

def extract_parameters(tag_list, line):
    tag_data = {}
    for tag in tag_list:
        result = regex.findall(f'{tag}'+r'''
            (?<rec> #capturing group rec
            \( #open parenthesis
            (?: #non-capturing group
            [^()]++ #anyting but parenthesis one or more times without backtracking
            | #or
            (?&rec) #recursive substitute of group rec
            )*
            \) #close parenthesis
            )
            ''', line, flags=regex.VERBOSE)
        tag_data_tag = []
        for r in result:
            tdata = parseFnParams(r)
            tag_data_tag.append(tdata)
        if len(tag_data_tag) > 0:
            tag_data[tag] = tag_data_tag
    return tag_data
    
def add_conditional_parameters_with_variables(hdr_type, kvar, k, v, params=[], lines=[], conditional=False, constraint='iris_all', macros_flag=True):
    global valid_params, global_res
    def InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines):
        if conditional:
            appendConditionalParameters(expr, fdt+" "+fvar, params, lines, constraint, macros_flag)
        else:
            params.append(fdt+" "+fvar)
    arguments_start_index = find_start_index(v)
    i = arguments_start_index
    while (i<len(v)):
        f_param = remove_spaces(v[i])
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = v[i]
        i = i+1
        fvar = f_details[0]
        fdt = ''
        if len(f_details) > 1:
            fdt = f_details[1]
        if f_param == 'PARAM' and len(f_details)>2:
            expr = getPyExprString(f_details[2])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'PARAM_CONST' and len(f_details)>3:
            expr = getPyExprString(f_details[2])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'VEC_PARAM' and len(f_details)>2:
            fdt  = fdt + f" * "
            expr = getPyExprString(f_details[2])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'IN_TASK':
            if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
                fdt  = 'iris_mem' 
            expr = getPyExprString(f_details[-1])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'IN_TASK_OFFSET':
            if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
                fdt  = 'iris_mem' 
            expr = getPyExprString(f_details[-1])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'OUT_TASK':
            if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
                fdt  = 'iris_mem' 
            expr = getPyExprString(f_details[-1])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'OUT_TASK_OFFSET':
            if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
                fdt  = 'iris_mem' 
            expr = getPyExprString(f_details[-1])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'IN_OUT_TASK':
            if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
                fdt  = 'iris_mem' 
            expr = getPyExprString(f_details[-1])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'IN_OUT_TASK_OFFSET':
            if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
                fdt  = 'iris_mem' 
            expr = getPyExprString(f_details[-1])
            InsertConditionalDeclarationsWithVariable(expr, fdt, fvar, params, lines)
            one_param_exists = True
        elif f_param == 'VS_CODE_GEN':
            None # Do nothing
        elif f_param == 'TILED_CODE_GEN':
            None # Do nothing
        elif f_param == 'CPP_TASK_TEMPLATE':
            None # Do nothing
        else:
            params.append(fdt+f" "+" "+fvar)
            one_param_exists = True
    if len(params) > 0:
         lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params))
    return lines

def generateTaskAPIs(args, cu_type_lines, header_data_hash):
    global valid_params, global_res
    def replace_datatypes(param, dtype_hash):
        for k, v in dtype_hash.items():
            param = re.sub(r'\b'+k+r'\b', v, param)
        return param
    def get_headers(f_details, cu_type='iris_cpu', dtype_replace={}, start_index=1):
        include_headers = []
        vs_params = extract_parameters(['INCLUDE', 'EXPR', 'DIM', 'ZIP', 'LOOP', 'GENERIC', 'MAP', 'CFUNC', 'LOOP', 'FULL_SIZE', 'TILE_SIZE', 'EXCLUDE'], ", ".join(f_details[start_index:]))
        for k, vs_details in vs_params.items():
            if k == 'INCLUDE':
                include_headers += [ f"#include <{f}>" for f in vs_details[0] ]
        arguments, arguments_variable_hash = get_parameter_types(v, constraint=cu_type)
        params = (arguments_variable_hash.keys())
        params_decl = []
        for arg_data in arguments:
            p = replace_datatypes(arg_data[2], dtype_replace) + " " + arg_data[0]
            params_decl.append(p)
        expression_stmts = []
        for k, vs_details in vs_params.items():
            if k == 'EXPR':
                for exprs in vs_details:
                    for expr in exprs:
                        expr_str = re.sub(r'\[\]\[\]\[\]', '[__id][__jd][__kd]', expr)
                        expr_str = re.sub(r'\[\]\[\]', '[__id][__jd]', expr)
                        expr_str = re.sub(r'\[\]', '[__id]', expr)
                        expression_stmts.append(expr_str)
        return vs_params, include_headers, params, params_decl, arguments, arguments_variable_hash, expression_stmts
    def generate_cuda_kernel(kvar, k, lines, v, f_details):
        func_name = kvar
        dtype_conversions={'int32_t' : 'int',
                           'uint32_t' : 'unsigned int',
                           'int16_t'  : 'short',
                           'uint16_t'  : 'unsigned short',
                           'int8_t'  : 'char',
                           'uint8_t'  : 'unsigned char',
                           'int64_t'  : 'long long',
                           'uint64_t'  : 'unsigned long long'}
        vs_params, include_headers, params, params_decl, arguments, arguments_variable_hash, expression_stmts = get_headers(f_details, 'iris_cuda', dtype_conversions)
        include_headers_str = "\n".join(include_headers) 
        expressions_str = "\n          ".join(expression_stmts)
        params_decl = ",\n".join(params_decl)
        params_str = ", ".join(params)
        lines.append(f"""
{include_headers_str}
extern \"C\" __global__ void {func_name}(
    {params_decl})
{{
    size_t __id =  blockIdx.x * blockDim.x + threadIdx.x;;
    {expressions_str};
}}
                """)
    def generate_hip_kernel(kvar, k, lines, v, f_details):
        func_name = kvar 
        dtype_conversions={'int32_t' : 'int',
                           'uint32_t' : 'unsigned int',
                           'int16_t'  : 'short',
                           'uint16_t'  : 'unsigned short',
                           'int8_t'  : 'char',
                           'uint8_t'  : 'unsigned char',
                           'int64_t'  : 'long long',
                           'uint64_t'  : 'unsigned long long'}
        vs_params, include_headers, params, params_decl, arguments, arguments_variable_hash, expression_stmts = get_headers(f_details, 'iris_hip', dtype_conversions)
        include_headers_str = "\n".join(include_headers) 
        expressions_str = "\n          ".join(expression_stmts)
        params_decl = ",\n".join(params_decl)
        params_str = ", ".join(params)
        lines.append(f"""
#include <hip/hip_runtime.h>
{include_headers_str}
extern \"C\" __global__ void {func_name}(
    {params_decl})
{{
    size_t __id =  blockIdx.x * blockDim.x + threadIdx.x;;
    {expressions_str};
}}
                """)
    def generate_openmp_kernel(kvar, k, lines, v, f_details):
        func_name = kvar
        vs_params, include_headers, params, params_decl, arguments, arguments_variable_hash, expression_stmts = get_headers(f_details, 'iris_cpu')
        include_headers_str = "\n".join(include_headers) 
        expressions_str = "\n          ".join(expression_stmts)
        params_decl = ",\n".join(params_decl)
        params_str = ", ".join(params)
        lines.append(f"""
{include_headers_str}
extern \"C\" void {func_name}(
    {params_decl}, IRIS_OPENMP_KERNEL_ARGS)
{{
    size_t __id;
    #pragma omp parallel for shared({params_str}) private(__id)
    IRIS_OPENMP_KERNEL_BEGIN(__id)
        {expressions_str};
    IRIS_OPENMP_KERNEL_END
}}
                """)
    def generate_tiled_code(cpp_api_name, lines, v, data_details_hash, f_details, cpp_task_template):
        vs_params, include_headers, params, params_decl, arguments, arguments_variable_hash, expression_stmts = get_headers(f_details, 'iris_all', start_index=0)
        include_headers_str = "\n".join(include_headers) 
        expressions_str = "\n          ".join(expression_stmts)
        params_str = ", ".join(params)
        loop_stmt = ""
        tiling_ds = []
        param_map = vs_params['MAP']
        param_map_hash = {}
        for p in param_map:
            p1 = p[1]
            param_map_hash[p[0]] = p[1]
        g_func_name = vs_params['GENERIC'][0][0]
        c_func_name = vs_params['CFUNC'][0][0]
        tile_sizes = vs_params['TILE_SIZE']
        full_sizes = vs_params['FULL_SIZE']
        excludes = [] 
        if 'EXCLUDE' in vs_params:
            excludes = vs_params['EXCLUDE'][0]
        zip_params = vs_params['ZIP']
        dim = int(vs_params['DIM'][0][0])
        zip_ds = []
        zip_ds_read = []
        zip_ds_decl = []
        zip_ds_iter = []
        zip_param_types = [ f"" ]
        full_sizes_str = ", ".join([ f"{x}" for x in full_sizes[0] ])
        tile_sizes_str = ", ".join([ f"{x}" for x in tile_sizes[0] ])
        params_decl = []
        params_decl.append("int target_dev")
        params_decl.append("iris_graph graph")
        cpp_params_decl = params_decl + []
        cpp_params_vars = [ "target_dev", "graph" ] 
        zip_index = 0
        zip_args = []
        zip_out_args = []
        cpp_call_args = ["task"]
        for index, arg_data in enumerate(arguments):
            x = arg_data[0]
            f_param = arg_data[1]
            vector_f_param = False
            if f_param=="IN_TASK" or f_param=="IN_TASK_OFFSET" or f_param =="OUT_TASK" or f_param == "IN_OUT_TASK" or f_param =="OUT_TASK_OFFSET" or f_param == "IN_OUT_TASK_OFFSET":
                vector_f_param = True
            vector_out_f_param = False
            if f_param =="OUT_TASK" or f_param == "IN_OUT_TASK" or f_param =="OUT_TASK_OFFSET" or f_param == "IN_OUT_TASK_OFFSET":
                vector_out_f_param = True
            if vector_f_param:
                cpp_call_args.append(f"{x}_tile.IRISMem()")
            else:
                cpp_call_args.append(x)
            if x in excludes:
                continue
            x_type = arg_data[3]
            params_decl.append(arg_data[2]+" "+x)
            arg_hash = arguments_variable_hash[x]
            if vector_f_param:
                cpp_params_decl.append(f"Tiling{dim}D<{x_type}> & {x}_tiling")
                cpp_params_vars.append(f"{x}_tiling")
                if vector_out_f_param:
                    zip_out_args.append(x)
                zip_args.append(f"{x}_tile")
                zip_ds.append(f"{x}_tiling")
                zip_ds_decl.append(f"Tiling{dim}D<{x_type}> {x}_tiling({x}, {full_sizes_str}, {tile_sizes_str});")
                zip_ds_iter.append(f"{x}_tiling.zitems()")
                zip_param_types.append(f"{x_type}")
                zip_ds_read.append(f"Tile{dim}D<{x_type}> {x}_tile = std::get<{zip_index}>(Z);")
                zip_index += 1
            else:
                cpp_params_decl.append(f"{x_type} {x}")
                cpp_params_vars.append(f"{x}")
        for index, cf in enumerate(cpp_call_args):
            if cf in param_map_hash:
                cf = param_map_hash[cf]
            zds = zip_args[0]
            cf = re.sub(r'row_tile_size', f"{zds}.row_tile_size", cf)
            cf = re.sub(r'col_tile_size', f"{zds}.col_tile_size", cf)
            cpp_call_args[index] = cf

        loop_stmt = "for(auto && Z : zip("+", ".join(zip_ds_iter)+"))"
        params_decl = params_decl + [ f"size_t {x}" for x in vs_params['FULL_SIZE'][0] ]
        params_decl = params_decl + [ f"size_t {x}" for x in vs_params['TILE_SIZE'][0] ]
        params_decl_str = ",\n       ".join(params_decl)
        cpp_params_decl_str = ",\n       ".join(cpp_params_decl)
        cpp_params_vars_str = ",\n       ".join(cpp_params_vars)
        zip_ds_decl_str = "\n".join(zip_ds_decl)
        zip_ds_read_str = "\n".join(zip_ds_read)
        task_set_metadata = []
        if dim == 2:
            task_set_metadata = [ 
                f"iris_task_set_metadata(task, 0, {zip_args[0]}.row_tile_index());",
                f"iris_task_set_metadata(task, 1, {zip_args[0]}.col_tile_index());"     ]
        elif dim == 1:
            task_set_metadata = [ 
                f"iris_task_set_metadata(task, 0, {zip_args[0]}.row_tile_index());"     ]
        task_set_metadata_str = "\n".join(task_set_metadata)
        flush_cmds = []
        for x in zip_out_args:
            flush_cmds.append(f"iris_task_dmem_flush_out(task, {x}_tile.IRISMem());")
        flush_cmds_str = "\n".join(flush_cmds)
        cpp_call_fn_name = cpp_api_name
        if 'CPP_TASK_TEMPLATE' in data_details_hash:
            cpp_task_template = extract_parameters(['MAP'], data_details_hash['CPP_TASK_TEMPLATE'][1])
            cpp_call_fn_name = data_details_hash['CPP_TASK_TEMPLATE'][0]+"<"+cpp_task_template['MAP'][0][0]+">"
        cpp_call = cpp_call_fn_name+"("+", ".join(cpp_call_args)+")"
        lines.append(f"""
{include_headers_str}
#ifndef IRIS_API_DEFINITION
#ifdef __cplusplus
void {g_func_name}({cpp_params_decl_str});
void {g_func_name}({params_decl_str});
#endif // __cplusplus
#ifdef __cplusplus
extern \"C\"  {{
#endif
void {c_func_name}({params_decl_str});
#ifdef __cplusplus
}}
#endif
#else //IRIS_API_DEFINITION
#ifdef __cplusplus
void {g_func_name}({cpp_params_decl_str})
{{
    {loop_stmt} {{
        {zip_ds_read_str}
        {expressions_str};
        iris_task task;
        iris_task_create_name(NULL, &task);
        {cpp_call};
        {flush_cmds_str};
        {task_set_metadata_str};
        iris_graph_task(graph, task, target_dev, NULL);
    }}
}}
void {g_func_name}({params_decl_str})
{{
    {zip_ds_decl_str}
    {g_func_name}({cpp_params_vars_str});
}}
#endif // __cplusplus
#ifdef __cplusplus
extern \"C\"  {{
#endif
void {c_func_name}({params_decl_str})
{{
    {zip_ds_decl_str}
    {g_func_name}({cpp_params_vars_str});
}}
#ifdef __cplusplus
}}
#endif
#endif //IRIS_API_DEFINITION
                """)
    def generate_cpp_task_template_code(cpp_api_name, c_api_name, kvar_original, lines, v, f_details, kernel_apis):
        vs_params, include_headers, params, params_decl, arguments, arguments_variable_hash, expression_stmts = get_headers(f_details, 'iris_all', start_index=0)
        include_headers_str = "\n".join(include_headers) 
        expressions_str = "\n          ".join(expression_stmts)
        params_str = ", ".join(params)
        func_name = f_details[0]
        param_map = vs_params['MAP']
        param_map_src_dtype = param_map[0][0]
        param_map_dtype = param_map[0][1]
        cpp_call_args = ["task"]
        task_api_call_args = ["task"]
        cpp_call_args_irismem = ["target_dev"]
        params_decl_irismem  = ["iris_task task"]
        params_templ_decl_irismem  = ["iris_task task"]
        params_decl = ["int target_dev"]
        params_decl_template = ["int target_dev"]
        params_templ_decl_template = ["int target_dev"]
        c_core_args = ["target_dev"]
        c_task_args = ["task"]
        cpp_params_decl = []
        unique_macro = []
        out_mems = []
        all_mems = []
        task_params = []
        task_params_info = []
        task_params_device_map = []
        access_data = {
            'IN_TASK' : "iris_r",
            'IN_TASK_OFFSET' : "iris_r",
            'OUT_TASK' : "iris_w",
            'OUT_TASK_OFFSET' : "iris_w",
            'IN_OUT_TASK' : "iris_rw",
            'IN_OUT_TASK_OFFSET' : "iris_rw"
        }
        for index, arg_data in enumerate(arguments):
            x = arg_data[0]
            f_param = arg_data[1]
            vector_f_param = False
            if f_param=="IN_TASK" or f_param=="IN_TASK_OFFSET" or f_param =="OUT_TASK" or f_param == "IN_OUT_TASK" or f_param =="OUT_TASK_OFFSET" or f_param == "IN_OUT_TASK_OFFSET":
                vector_f_param = True
            vector_out_f_param = False
            if f_param =="OUT_TASK" or f_param == "IN_OUT_TASK" or f_param =="OUT_TASK_OFFSET" or f_param == "IN_OUT_TASK_OFFSET":
                vector_out_f_param = True
            if vector_out_f_param:
                out_mems.append(x)
            c_core_args.append(x)
            c_task_args.append(x)
            params_decl.append(arg_data[2]+" "+x)
            x_template = arg_data[2]
            task_params_device_map.append(arg_data[4])
            templ_x = re.sub(r'\b'+param_map_src_dtype+r'\b', param_map_dtype, arg_data[2])
            if vector_f_param:
                all_mems.append((x, arg_data[5][4]))
                task_params.append(f"&{x}")
                task_params_info.append(access_data[f_param])
                x_template = re.sub(r'\b'+param_map_src_dtype+r'\b', param_map_dtype, x_template)
                cpp_call_args.append(f"{x}")
                task_api_call_args.append(f"__iris_{x}")
                cpp_call_args_irismem.append(f"{x}")
                params_decl_irismem.append("iris_mem "+x)
                params_templ_decl_irismem.append("iris_mem "+x)
                unique_macro.append(x)
            else:
                task_params.append(f"&{x}")
                task_params_info.append(f"sizeof({x})")
                cpp_call_args.append(x)
                task_api_call_args.append(f"{x}")
                cpp_call_args_irismem.append(f"{x}")
                params_decl_irismem.append(arg_data[2]+" "+x)
                params_templ_decl_irismem.append(templ_x+" "+x)
            params_decl_template.append(x_template+" "+x)
            params_templ_decl_template.append(templ_x+" "+x)
        flush_cmds = []
        for x in out_mems:
            flush_cmds.append(f"iris_task_dmem_flush_out(task, __iris_{x});")
        flush_cmds_str = "\n".join(flush_cmds)
        params_decl_template_str = ",\n       ".join(params_decl_template)
        params_templ_decl_template_str = ",\n       ".join(params_templ_decl_template)
        params_decl_str = ",\n       ".join(params_decl)
        params_decl_irismem_str = ",\n       ".join(params_decl_irismem)
        params_templ_decl_irismem_str = ",\n       ".join(params_templ_decl_irismem)
        cpp_call_args_irismem_str = ",\n       ".join(cpp_call_args_irismem)
        cpp_call_args_str = ",\n       ".join(cpp_call_args)
        unique_macro_str = f"{func_name}_"+"_".join(unique_macro)
        unique_macro_str = unique_macro_str.upper()
        unique_macro_str = re.sub(r'\(', '__', unique_macro_str)
        unique_macro_str = re.sub(r'\)', '__', unique_macro_str)
        unique_macro_str = re.sub(r',', '_', unique_macro_str)
        unique_macro_str = re.sub(r' ', '_', unique_macro_str)
        c_task_api_name = kernel_apis[1]
        c_core_api_name = kernel_apis[0]
        task_params_str = ", ".join(task_params)
        task_params_info_str = ", ".join(task_params_info)
        task_params_device_map_str = ", ".join(task_params_device_map)
        c_core_args_str = ",\n      ".join(c_core_args)
        c_task_args_str = ",\n      ".join(c_task_args)
        offset = v[3]
        gws = v[4]
        lws = v[5]
        if gws == 'NULL_LWS':
            gws = "(size_t *)NULL"
        else:
            gws_params = extract_parameters(['GWS'], gws)
            gws = "{"+", ".join(gws_params['GWS'][0])+"}"
        if lws == 'NULL_LWS':
            lws = "(size_t *)NULL"
        else:
            lws_params = extract_parameters(['LWS'], lws)
            lws = "{"+", ".join(lws_params['LWS'][0])+"}"
        if offset == 'NULL_OFFSET':
            offset = "(size_t *)NULL"
        else:
            offset_params = extract_parameters(['OFFSET'], offset)
            offset = "{"+", ".join(offset_params['OFFSET'][0])+"}"
        kernel_name = kvar_original
        kernel_name = re.sub(r'\bIRIS_FUNC\b', 'IRIS_FUNC_STR', kernel_name)
        dim = v[2]
        iris_mem_create = [ f"iris_mem __iris_{x[0]}; iris_data_mem_create(&__iris_{x[0]}, {x[0]}, {x[1]});" for x in all_mems ]
        iris_mem_create_stmts = "\n    ".join(iris_mem_create)
        iris_mem_release_stmts = "\n    ".join([f"iris_mem_release(__iris_{x[0]});" for x in all_mems])
        iris_mem_flush_stmts = flush_cmds_str
        task_api_call_args_str = ",\n       ".join(task_api_call_args)
        lines.append(f"""
#pragma once
#ifdef __cplusplus
#ifndef {unique_macro_str}
#define {unique_macro_str}
template <typename {param_map_dtype}>
void {func_name}({params_templ_decl_irismem_str})
{{
}}
template <typename {param_map_dtype}>
void {func_name}({params_templ_decl_template_str})
{{
}}
#endif // {unique_macro_str}
#ifndef IRIS_API_DEFINITION
#ifdef __cplusplus
extern "C" {{
#endif //__cplusplus
void {c_task_api_name}({params_decl_irismem_str});
void {c_core_api_name}({params_decl_str});
#ifdef __cplusplus
}}
#endif //__cplusplus
#ifdef __cplusplus
void {cpp_api_name}({params_decl_irismem_str});
void {cpp_api_name}({params_decl_str});
template <>
void {func_name}<{param_map_src_dtype}>({params_decl_irismem_str})
{{
    {cpp_api_name}({c_task_args_str});
}}
template <>
void {func_name}<{param_map_src_dtype}>({params_decl_str})
{{
    {cpp_api_name}({c_core_args_str});
}}
#endif
#else //IRIS_API_DEFINITION               
#ifdef __cplusplus
extern "C" {{
#endif //__cplusplus
void {c_task_api_name}({params_decl_irismem_str})
{{
    void* __task_params[] = {{ {task_params_str} }};
    int __task_params_info[] = {{ {task_params_info_str} }};
    int __task_params_device_map[] = {{ {task_params_device_map_str} }};
    size_t *__st_offset = {offset};
    size_t __st_gws[] = {gws};
    size_t *__st_lws = {lws};
    iris_task_kernel(task, "{kernel_name}", {dim}, __st_offset, __st_gws, __st_lws, sizeof(__task_params_info)/sizeof(int), __task_params, __task_params_info);
    iris_params_map(task, __task_params_device_map);
}}
#ifdef __cplusplus
}}
#endif //__cplusplus
#ifdef __cplusplus
extern "C" {{
#endif //__cplusplus
void {c_core_api_name}({params_decl_str})
{{
    iris_task task;
    iris_task_create(&task);
    {iris_mem_create_stmts}
    {c_task_api_name}({task_api_call_args_str});
    {iris_mem_flush_stmts}
    iris_task_submit(task, target_dev, NULL, 1);
    {iris_mem_release_stmts}
}}
#ifdef __cplusplus
}}
#endif //__cplusplus
void {cpp_api_name}({params_decl_irismem_str})
{{
    {c_task_api_name}({c_task_args_str});
}}
void {cpp_api_name}({params_decl_str})
{{
    {c_core_api_name}({c_core_args_str});
}}
template <>
void {func_name}<{param_map_src_dtype}>({params_decl_irismem_str})
{{
    {cpp_api_name}({c_task_args_str});
}}
template <>
void {func_name}<{param_map_src_dtype}>({params_decl_str})
{{
    {cpp_api_name}({c_core_args_str});
}}
#endif // __cplusplus
#endif // IRIS_API_DEFINITION
                """)
    def insert_code(data_hash, key, lines):
        if key not in data_hash:
            data_hash[key] = []
        data_hash[key] = data_hash[key] + lines
    # VS_CODE_GEN(1, iris_cpu, (math.h), #3 = func(#1, #2))
    for cpp_k, (hdr_type, v, data_details_hash, code, kvar_original, kernel_apis) in header_data_hash.items():
        if code != IRIS_CREATE_APIS or hdr_type != CPP_CORE_API:
            continue
        func_name = v[1]
        (cpp_api_name, c_api_name) = cpp_k
        kvar = getKernelParamVairable(func_name)
        func_sig = "int "+func_name+"("
        if hdr_type == C_CORE_API or hdr_type == C_TASK_API:
            func_sig = f"""
#ifdef __cplusplus
extern \"C\" 
#endif
int {func_name}("""
        arguments_start_index = find_start_index(v)
        i = arguments_start_index
        cpp_task_template = {}
        while (i<len(v)):
            f_param = remove_spaces(v[i])
            i = i+1
            if f_param not in valid_params:
                continue
            f_details = v[i]
            i = i+1
            fvar = f_details[0]
            fdt = ''
            if len(f_details) > 1:
                fdt = f_details[1]
            if f_param == 'VS_CODE_GEN':
                lines = []
                if f_details[0] == 'iris_cpu':
                    generate_openmp_kernel(kvar_original, c_api_name, lines, v, f_details)
                elif f_details[0] == 'iris_cuda':
                    generate_cuda_kernel(kvar_original, c_api_name, lines, v, f_details)
                elif f_details[0] == 'iris_hip':
                    generate_hip_kernel(kvar_original, c_api_name, lines, v, f_details)
                elif f_details[0] == 'iris_opencl':
                    generate_opencl_kernel(kvar_original, c_api_name, lines, v, f_details)
                insert_code(cu_type_lines, f_details[0], lines)
            elif f_param == 'TILED_CODE_GEN':
                lines = []
                lines.append("#pragma once")
                generate_tiled_code(cpp_api_name, lines, v, data_details_hash, f_details, cpp_task_template)
                insert_code(cu_type_lines, "tiled", lines)
            elif f_param == 'CPP_TASK_TEMPLATE':
                lines = []
                generate_cpp_task_template_code(cpp_api_name, c_api_name, kvar_original, lines, v, f_details, kernel_apis)
                insert_code(cu_type_lines, "cpp_task_template", lines)
    def write_kernel_to_file(cu_type, filename):
        if cu_type in cu_type_lines and len(cu_type_lines[cu_type])>0:
            print(f"Writing {cu_type} kernels to file: {filename}")
            lines = [ """
/**
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */
            """]
            lines += cu_type_lines[cu_type]
            with open(filename, 'w') as fh:
                fh.write("\n".join(lines))
                fh.close()
    write_kernel_to_file('cpp_task_template', 'codegen_task_template.h')
    write_kernel_to_file('tiled', 'codegen_tiled.h')
    write_kernel_to_file('iris_cpu', 'codegen_openmp_kernels.h')
    write_kernel_to_file('iris_cuda', 'codegen_cuda_kernels.cu')
    write_kernel_to_file('iris_hip', 'codegen_hip_kernels.hip.cpp')

def extract_header_data_details_hash(args, v):
    global valid_params, global_res
    arguments_start_index = find_start_index(v)
    i = arguments_start_index
    f_details_hash = {}
    while (i<len(v)):
        f_param = remove_spaces(v[i])
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = v[i]
        i = i+1
        fvar = f_details[0]
        fdt = ''
        if len(f_details) > 1:
            fdt = f_details[1]
        f_details_hash[f_param] = f_details
    return f_details_hash

def appendKernelSignatureHeaderFile(args, lines, header_data_hash):
    global valid_params, global_res
    for hdr in args.signature_file_include_headers:
        hdr_list = hdr.split(",")
        for each_hdr in hdr_list:
            each_hdr_str = remove_spaces(each_hdr)
            lines.append(f"#include \"{each_hdr_str}\"")
    for k, (hdr_type, v, data_details_hash, code, kvar_original, kernel_apis) in header_data_hash.items():
        func_name = k
        if hdr_type == CPP_CORE_API or hdr_type == CPP_TASK_API:
            func_name = k[0]
            lines.append("#ifdef __cplusplus")
        kvar = getKernelParamVairable(func_name)
        func_sig = "int "+func_name+"("
        if hdr_type == C_CORE_API or hdr_type == C_TASK_API:
            func_sig = f"""
#ifdef __cplusplus
extern \"C\" 
#endif
int {func_name}("""
        params = []
        if hdr_type == C_TASK_API or hdr_type == CPP_TASK_API:
            params.append("iris_task task")
        else:
            params.append("int target_dev")
        lines.append(func_sig)
        add_conditional_parameters_with_variables(hdr_type, kvar, k, v, params, lines)
        lines.append(");")
        if hdr_type == CPP_CORE_API or hdr_type == CPP_TASK_API:
            lines.append("#endif // __cplusplus")

def appendKernelSignature(args, lines, data_hash, k_hash):
    global valid_params, global_res
    def InsertConditionalDeclarations(k, expr, fdt, params, lines):
        appendConditionalParameters(expr, fdt, params, lines)
    for k,v in data_hash.items():
        kvar = getKernelParamVairable(k)
        func_sig = "int iris_kernel_"+k+"(HANDLETYPE"
        if args.thread_safe:
            func_sig = "typedef int (* __iris_kernel_"+k+"_ptr)(HANDLETYPE DEVICE_NUM_TYPE"
        params = []
        lines.append(func_sig)
        arguments_start_index = find_start_index(v)
        i = arguments_start_index
        while (i<len(v)):
            f_param = remove_spaces(v[i])
            i = i+1
            if f_param not in valid_params:
                continue
            f_details = v[i]
            i = i+1
            fvar = f_details[0]
            fdt = ''
            if len(f_details) > 1:
                fdt = f_details[1]
            if f_param == 'PARAM' and len(f_details)>2:
                expr = getPyExprString(f_details[2])
                InsertConditionalDeclarations(k, expr, fdt+f" /*{fvar}*/", params, lines)
                one_param_exists = True
            elif f_param == 'PARAM_CONST' and len(f_details)>3:
                expr = getPyExprString(f_details[2])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                one_param_exists = True
            elif f_param == 'VEC_PARAM' and len(f_details)>2:
                fdt  = fdt + f" * /*{fvar}*/"
                expr = getPyExprString(f_details[2])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                one_param_exists = True
            elif f_param == 'IN_TASK':
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\tint,")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'IN_TASK_OFFSET':
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\tint,")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'OUT_TASK':
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\tint,")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'OUT_TASK_OFFSET':
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\tint,")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'IN_OUT_TASK':
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\tint,")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'IN_OUT_TASK_OFFSET':
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarations(k, expr, fdt, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\tint,")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'VS_CODE_GEN':
                None
            elif f_param == 'TILED_CODE_GEN':
                None
            elif f_param == 'CPP_TASK_TEMPLATE':
                None
            else:
                params.append(fdt+f" /*{fvar}*/")
                one_param_exists = True
        if len(params) > 0:
             lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
        lines.append("#ifndef ENABLE_IRIS_HEXAGON_APIS")
        lines.append("size_t, size_t")
        lines.append("#else //ENABLE_IRIS_HEXAGON_APIS")
        lines.append("int, int")
        lines.append("#endif //ENABLE_IRIS_HEXAGON_APIS")
        lines.append(");")

def appendSetKernelWrapperFunctions(args, lines, data_hash, k_hash):
    lines.append('''
void iris_set_kernel_ptr_with_obj(void *obj, __iris_kernel_ptr ptr) {
    iris_base_pack *bobj = (iris_base_pack *)obj;
    bobj->__iris_kernel = ptr;
}
            ''')

def appendHandleWrapperFunctions(args, lines, data_hash, k_hash):
    lines.append('''
void iris_kernel_set_queue_with_obj(void *obj, void *queue) {
    iris_base_pack *bobj = (iris_base_pack *)obj;
    bobj->__iris_handle = queue;
}
void *iris_kernel_get_queue_with_obj(void *obj) {
    iris_base_pack *bobj = (iris_base_pack *)obj;
    return bobj->__iris_handle;
}
            ''')

def appendKernelNamesFunction(args, lines, data_hash, k_hash):
    lines.append("static char __iris_kernel_names[][256] = {")
    kernel_names = []
    for k,v in data_hash.items():
        kernel_names.append(f"\"{k}\"")
    kernel_names.append("\"\"")
    lines.append("\t"+",\n\t".join(kernel_names))
    lines.append("};")
    lines.append("char (*iris_get_kernel_names())[256] {")
    lines.append("\t return __iris_kernel_names;")
    lines.append("};")

def find_start_index(v):
    global global_arguments_start_index
    re_pattern = re.compile(r'^\s*IN_TASK|^\s*IN_OUT_TASK|^\s*OUT_TASK|^\s*PARAM|^\s*VS_CODE_GEN|^\s*TILED_CODE_GEN|^\s*CPP_TASK_TEMPLATE|^\s*VEC_PARAM')
    for index, tok in enumerate(v):
        if re_pattern.search(tok):
            return index
    return global_arguments_start_index

def appendStructure(args, lines, data_hash, k_hash):
    global valid_params
    lines.append("typedef void (*__iris_kernel_ptr)();")
    lines.append("typedef struct {")
    lines.append("__global int __iris_kernel_idx;")
    lines.append("__global void *__iris_handle;")
    lines.append("__global __iris_kernel_ptr __iris_kernel;")
    lines.append("} iris_base_pack;")
    for k,v in data_hash.items():
        kvar = getKernelParamVairable(k)
        lines.append("typedef struct {")
        params = []
        if args.thread_safe:
            params.append("__global int __iris_kernel_idx;")
            params.append("__global void *__iris_handle;")
            params.append("__global __iris_kernel_"+k+"_ptr __iris_kernel;")
        param_hash = {}
        arguments_start_index = find_start_index(v)
        i = arguments_start_index
        while (i<len(v)):
            f_param = remove_spaces(v[i])
            i = i+1
            if f_param not in valid_params:
                continue
            f_details = v[i]
            i = i+1
            if f_param in ['VS_CODE_GEN', 'CPP_TASK_TEMPLATE', 'TILED_CODE_GEN']:
                continue
            fvar = f_details[0]
            fdt = ''
            if len(f_details) > 1:
                fdt = f_details[1]
            if fvar not in param_hash:
                param_hash[fvar] = True
                if f_param == "VEC_PARAM":
                    params.append("__global "+fdt+" *"+fvar+";")
                else:
                    params.append("__global "+fdt+" "+fvar+";")
                if f_param == "IN_TASK" or f_param == "IN_TASK_OFFSET" or f_param == "OUT_TASK" or f_param == "OUT_TASK_OFFSET" or f_param == "IN_OUT_TASK" or f_param == "IN_OUT_TASK_OFFSET":
                    params.append("__global int "+fvar+"___size;")
        lines.append("\t"+"\n\t".join(params))
        lines.append("} iris_"+kvar+";")
        if args.thread_safe != 1:
            lines.append("static iris_"+kvar+" "+kvar+";")

def appendKernelSetArgMemParamFunctions(args, lines, kernel, data):
    global valid_params
    kvar = getKernelParamVairable(kernel)
    lines.append("static int "+kernel+"_setarg(void *obj, int idx, size_t size, void* value) {")
    lines.append('''
  switch (idx) {
    ''')
    p_index = 0
    arguments_start_index = find_start_index(data)
    i = arguments_start_index
    while (i<len(data)):
        f_param = remove_spaces(data[i])
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = data[i]
        i = i+1
        fvar = f_details[0]
        if f_param=="PARAM":
            lines.append("\t\t\tcase "+str(p_index)+": memcpy(&((iris_"+kvar+" *)obj)->"+fvar+", value, size); break;")
        elif f_param=="PARAM_CONST":
            lines.append("\t\t\tcase "+str(p_index)+": memcpy(&((iris_"+kvar+" *)obj)->"+fvar+", value, size); break;")
        p_index = p_index+1
    lines.append('''
        default: return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}
        ''')
    lines.append("static int "+kernel+"_setmem(void *obj, int idx, void *mem, int size) {")
    lines.append('''
  switch (idx) {
    ''')
    p_index = 0
    i = arguments_start_index
    while (i<len(data)):
        f_param = remove_spaces(data[i])
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = data[i]
        i = i+1
        fvar = f_details[0]
        fdt = ''
        if len(f_details) > 1:
            fdt = f_details[1]
        if f_param=="IN_TASK" or f_param=="IN_TASK_OFFSET" or f_param =="OUT_TASK" or f_param == "IN_OUT_TASK" or f_param =="OUT_TASK_OFFSET" or f_param == "IN_OUT_TASK_OFFSET":
            lines.append("\t\t\tcase "+str(p_index)+": ((iris_"+kvar+" *)obj)->"+fvar+" = ("+fdt+")mem; ((iris_"+kvar+" *)obj)->"+fvar+"___size = size; break;")
        elif f_param=="VEC_PARAM":
            lines.append("\t\t\tcase "+str(p_index)+": ((iris_"+kvar+" *)obj)->"+fvar+" = ("+fdt+"*)mem; break;")
        p_index = p_index+1
    lines.append('''
        default: return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}
        ''')

def appendKernelSetArgMemGlobalFunctions(args, lines, data_hash, k_hash):
    if args.thread_safe == 1:
        lines.append('''
int iris_setarg_with_obj(void *obj, int idx, size_t size, void* value) {
  int kernel_idx = *((int *)obj); // First argument should be kernel index
  switch (kernel_idx) {
        ''')
        for k,v in data_hash.items():
            lines.append("\t\t\t case "+str(k_hash[k])+": return "+k+"_setarg(obj, idx, size, value);")
        lines.append("\t\t\t default: break;");
        lines.append('''
    }
    return IRIS_ERROR;
}
        ''')
        lines.append('''
int iris_setmem_with_obj(void *obj, int idx, void *mem, int size) {
  int kernel_idx = *((int *)obj); // First argument should be kernel index
  switch (kernel_idx) {
    ''')
        for k,v in data_hash.items():
            lines.append("\t\t\t case "+str(k_hash[k])+": return "+k+"_setmem(obj, idx, mem, size);")
        lines.append("\t\t\t default: break;");
        lines.append('''
    }
    return IRIS_ERROR;
}
        ''')
    else:
        lines.append('''
int iris_setarg(int idx, size_t size, void* value) {
  switch (iris_kernel_idx) {
    ''')
        for k,v in data_hash.items():
            kvar = getKernelParamVairable(k)
            lines.append("\t\t\t case "+str(k_hash[k])+": return "+k+"_setarg((void *)&"+kvar+", idx, size, value);")
        lines.append("\t\t\t default: break;");
        lines.append('''
    }
    return IRIS_ERROR;
}
        ''')
        lines.append('''
int iris_setmem(int idx, void *mem, int size) {
  switch (iris_kernel_idx) {
    ''')
        for k,v in data_hash.items():
            kvar = getKernelParamVairable(k)
            lines.append("\t\t\t case "+str(k_hash[k])+": return "+k+"_setmem((void *)&"+kvar+", idx, mem, size);")
        lines.append("\t\t\t default: break;");
        lines.append('''
    }
    return IRIS_ERROR;
}
        ''')

def appendKernelLaunchFunction(args, lines, data_hash, k_hash):
    global valid_params
    if args.thread_safe == 1:
        lines.append('''
int iris_kernel_with_obj(void *obj, const char* name) {
    int *kernel_idx_p = ((int *)obj); // First argument should be kernel index
    ''')
        for k,v in data_hash.items():
            kvar = getKernelParamVairable(k)
            lines.append("\t if (strcmp(name, \""+k+"\") == 0) {")
            lines.append("\t\t *kernel_idx_p = "+str(k_hash[k])+";")
            lines.append("\t\t return IRIS_SUCCESS;")
            lines.append("\t }")
        lines.append("\t *kernel_idx_p = -1;")
        lines.append('''
    return IRIS_ERROR;
}
        ''')
    else:
        lines.append('''
int iris_kernel(const char* name) {
    iris_kernel_lock();
    ''')
        for k,v in data_hash.items():
            lines.append("\t if (strcmp(name, \""+k+"\") == 0) {")
            lines.append("\t\t iris_kernel_idx = "+str(k_hash[k])+";")
            lines.append("\t\t return IRIS_SUCCESS;")
            lines.append("\t }")
        lines.append("\t iris_kernel_idx = -1;")
        lines.append('''
    return IRIS_ERROR;
}
        ''')

    def InsertKernelLaunchSwith(lines, data_hash, is_with_obj=False):
        if is_with_obj:
            lines.append('''
        int kernel_idx = *((int *)obj); // First argument should be kernel index
        switch(kernel_idx) {
        ''')
        else:
            lines.append('''
        switch(iris_kernel_idx) {
        ''')
        handle_var = 'HANDLE'
        if is_with_obj:
            handle_var = 'HANDLE_WITH_OBJ'
        for k,v in data_hash.items():
            kvar = getKernelParamVairable(k)
            lines.append("\t\t\tcase "+str(k_hash[k])+": {")
            fn_name = "iris_kernel_"+k
            if is_with_obj:
                lines.append(f"\t\t\t  iris_{kvar} *kobj = (iris_{kvar}*)obj;")
                kvar = "kobj->"
                fn_name = "(kobj->__iris_kernel)"
            else:
                kvar = kvar+"."
            lines.append(f"\t\t\t  {fn_name}({handle_var} STREAM_VAR DEVICE_NUM")
            add_conditional_parameters_to_kernel(kvar, k, v, lines)
            lines.append("#ifndef DISABLE_OFFSETS")
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t(int)off, (int)ndr") 
            lines.append("#else")
            #lines.append("#ifdef ENABLE_IRIS_OPENMP_APIS")
            lines.append("\t\t\t\toff, ndr") 
            lines.append("#endif")
            lines.append("#endif")
            lines.append("\t\t\t  );");
            lines.append("\t\t\t  break;")
            lines.append("\t\t\t}");
        lines.append("\t\t\t default: break;");
        lines.append('''
          }
        ''')
    if args.thread_safe == 1:
        lines.append('''
int iris_launch_with_obj(
#ifdef IRIS_ENABLE_STREAMS
        StreamType  stream,
#endif
        void *obj,
        int devno, 
        int dim, size_t off, size_t ndr) {
        ''')
        InsertKernelLaunchSwith(lines, data_hash, True)
        lines.append('''
        return IRIS_SUCCESS;
}
        ''')
    else:
        lines.append('''
int iris_launch(
#ifdef IRIS_ENABLE_STREAMS
        StreamType  stream,
#endif
        int dim, size_t off, size_t ndr) {
        ''')
        InsertKernelLaunchSwith(lines, data_hash)
        lines.append('''
        iris_kernel_unlock();
        return IRIS_SUCCESS;
}
        ''')

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def InitParser(parser):
    parser.add_argument("-input", dest="input", nargs='+', default=[])
    parser.add_argument("-output", dest="output", default="iris_app_cpu_dsp_interface.h")
    parser.add_argument("-extract", dest="extract", action="store_true")
    parser.add_argument("-generate", dest="generate", action="store_true")
    parser.add_argument("-thread_safe", dest="thread_safe", type=int, default=1)
    parser.add_argument("-compile_options", dest="compile_options", default="")
    parser.add_argument("-signature_file", dest="signature_file", default="app.h")
    parser.add_argument("-signature_file_include_headers", dest="signature_file_include_headers", nargs='+', default=[])

def GetDefaultArgs():
    parser = argparse.ArgumentParser()
    InitParser(parser)
    args = parser.parse_args()
    return args 

def GetIRISTaskAPIsArgs(args=None, arg_string='', arg_dict={}):
    parser = argparse.ArgumentParser()
    InitParser(parser)
    if args == None:
        if arg_string != '':
            parser.parse_args(shlex.split(arg_string))
        elif arg_dict != None:
            args = GetDefaultArgs()
            for k,v in arg_dict.items():
                args.__dict__[k] = v
        else:
            args = parser.parse_args()
    return args

def ExtractIRISTaskAPIs(args=None, arg_string='', arg_dict={}, in_code_buffer='', write_output=True):
    args = GetIRISTaskAPIsArgs(args, arg_string, arg_dict)
    if not args.extract and not args.generate:
        args.extract = True 
    if args.extract and args.generate:
        if in_code_buffer != '':
            f = tempfile.NamedTemporaryFile(delete=False)
            fh = open(f.name+'.c', "w")
            fh.write(in_code_buffer)
            fh.close()
            print('Temporary file: ', f.name+'.c')
            args.__dict__['input'] = args.__dict__['input'] + [f.name+'.c']
        iris_calls = extractTaskCalls(args)
        output_lines = generateIrisInterfaceCode(args, iris_calls)
        if write_output:
            WriteFile(args.output, output_lines, tag="Consolidated CPU/DSP interface code for all kernels in header")
        if in_code_buffer != '':
            #os.unlink(f.name)
            None
    elif args.extract:
        iris_calls = extractTaskCalls(args)
        if write_output:
            WriteFile(args.output, iris_calls, tag='Generating metadata')
        output_lines = iris_calls
    elif args.generate:
        codes = read_metadata_file(args, args.input)
        output_lines = generateIrisInterfaceCode(args, codes)
        if write_output:
            WriteFile(args.output, output_lines, tag="Consolidated CPU/DSP interface code for all kernels in header")
    return output_lines

def test1():
    code = f'''
IRIS_CREATE_APIS(
        saxpy,
        saxpy_core,
        saxpy_task,
        "saxpy_kernel", 1,
        NULL_OFFSET, GWS(SIZE), NULL_LWS,
        OUT_TASK(Z, int32_t *, int32_t, Z, sizeof(int32_t)*SIZE),
        IN_TASK(X, const int32_t *, int32_t, X, sizeof(int32_t)*SIZE),
        IN_TASK(Y, const int32_t *, int32_t, Y, sizeof(int32_t)*SIZE),
        PARAM(SIZE, int32_t, iris_cpu),
        PARAM(A, int32_t),
        PARAM(cuUsecPtr, int32_t*, iris_dsp),
        PARAM(cuCycPtr, int32_t*, iris_dsp),
        VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(Z[] = func(X[], Y[], A)+func1(A, X[], Y[]))),
        VS_CODE_GEN(iris_cuda, INCLUDE(math.h, stdint.h), EXPR(
                Z[] = func(X[], Y[], A)+func1(A, X[], Y[])
                )),
        VS_CODE_GEN(iris_hip, INCLUDE(math.h, stdint.h), EXPR(Z[] = func(X[], Y[], A)+func1(A, X[], Y[])))
                    );
    '''
    ExtractIRISTaskAPIs(in_code_buffer=code, arg_dict={'extract': True, 'generate':True})

def test2():
    code = f'''
    IRIS_CREATE_APIS(
            matris_cosf,
            IRIS_FUNC(matris_core_, COS_TYPE),
            IRIS_FUNC(matris_task_, COS_TYPE),
            "matris_" IRIS_C_STR(cosf) "_kernel", 1,
            NULL_OFFSET, GWS(size), NULL_LWS,
            OUT_TASK(out, DTYPE*, DTYPE, out, sizeof(DTYPE)*size),
            IN_TASK(in, DTYPE*, DTYPE, in, sizeof(DTYPE)*size),
            PARAM(size, size_t, iris_cpu),
            CPP_TASK_TEMPLATE(matris_cos, MAP(DTYPE, DataType)),
            VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(out[] = COS_TYPE(in[]))),
            VS_CODE_GEN(iris_cuda, INCLUDE(math.h, stdint.h), EXPR(out[] = COS_TYPE(in[]))),
            VS_CODE_GEN(iris_hip, INCLUDE(math.h, stdint.h), EXPR(out[] = COS_TYPE(in[]))),
            TILED_CODE_GEN(FULL_SIZE(M), TILE_SIZE(tM), DIM(1), EXCLUDE(size),
                MAP(size, row_tile_size()),
                GENERIC(matris_tiled_cos),
                CFUNC(matris_tiled_cosf1d),
                ZIP(in,out)),
            TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(size),
                MAP(size, row_tile_size() * col_tile_size()),
                GENERIC(matris_tiled_cos),
                CFUNC(matris_tiled_cosf2d),
                ZIP(in, out))
    );

    '''
    ExtractIRISTaskAPIs(in_code_buffer=code, arg_dict={'extract': True, 'generate':True})
def test():
    #VS_CODE_GEN(iris_cpu, INCLUDE(math.h, stdlib.h, stdint.h), EXPR(Z[] = func(X[], Y[], A)+func1(A, X[], Y[]))),
    #VS_CODE_GEN(iris_cuda, INCLUDE(math.h, stdint.h), EXPR(
    #        Z[] = func(X[], Y[], A)+func1(A, X[], Y[])
    #        )),
    #VS_CODE_GEN(iris_hip, INCLUDE(math.h, stdint.h), EXPR(Z[] = func(X[], Y[], A)+func1(A, X[], Y[]))),
    code = f'''
IRIS_CREATE_APIS(
        saxpy_int,
        saxpy_core_int,
        saxpy_task_int,
        "saxpy_kernel", 1,
        NULL_OFFSET, GWS(SIZE), NULL_LWS,
        OUT_TASK(Z, int32_t *, int32_t, Z, sizeof(int32_t)*SIZE),
        IN_TASK(X, const int32_t *, int32_t, X, sizeof(int32_t)*SIZE),
        IN_TASK(Y, const int32_t *, int32_t, Y, sizeof(int32_t)*SIZE),
        PARAM(SIZE, int32_t, iris_cpu),
        PARAM(A, int32_t),
        CPP_TASK_TEMPLATE(saxpy, MAP(int32_t, DataType)),
        TILED_CODE_GEN(FULL_SIZE(M, N), TILE_SIZE(tM, tN), DIM(2), EXCLUDE(SIZE),
            MAP(SIZE, row_tile_size() * col_tile_size()),
            GENERIC(saxpy_tiled), 
            CFUNC(saxpy_tiled_int),  
            ZIP(X, Y, Z))
                    );
    '''
    ExtractIRISTaskAPIs(in_code_buffer=code, arg_dict={'extract': True, 'generate':True})

def main():
    ExtractIRISTaskAPIs()

if __name__ == "__main__":
    main()
    #test2()
