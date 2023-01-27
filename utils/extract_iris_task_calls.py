import re
import argparse
import pdb
import regex
import os
import tempfile

global_arguments_start_index=7
valid_params = { 'PARAM' : 1, 'PARAM_CONST' : 1, 'IN_TASK' : 1, 'IN_TASK_OFFSET' : 1, 'OUT_TASK_OFFSET' : 1, 'OUT_TASK' : 1, 'IN_OUT_TASK' : 1, 'IN_OUT_TASK_OFFSET' : 1, 'VEC_PARAM' : 1 }

global_res= { 
    "iris_all" : 0xFFFFFFFF, 
    "iris_cpu"    : 0x01, 
    "iris_dsp"    : 0x02, 
    "iris_fpga"   : 0x04, 
    "iris_gpu"    : 0x08,
    "iris_nvidia" : 0x10,
    "iris_amd"    : 0x20,
}

def remove_spaces(api_name):
    api_name = re.sub(r'"', '', api_name)
    api_name = re.sub(r'\s*', '', api_name)
    return api_name

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

def extractTaskCalls(args):
    file = args.input[0]
    f = tempfile.NamedTemporaryFile(delete=False)
    print(f"[Cmd] gcc -E {file} -Uiris_cpu -Uiris_dsp -o {f.name} {args.compile_options} -DUNDEF_IRIS_MACROS")
    os.system(f"gcc -E {file} -Uiris_cpu -Uiris_dsp -o {f.name} {args.compile_options} -DUNDEF_IRIS_MACROS")
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
    def add_code(result, code=0):
        result_out = [ '('+str(code)+', '+r[1:] for r in result ]
        return result_out
    fh.close()
    os.unlink(f.name)
    return add_code(result0)+add_code(result1)+add_code(result2)+add_code(result3)+add_code(result4, 1)

def writeExtracts(code, args):
    fh = open(args.output, "w")
    for l in code:
        fh.write(l+"\n")
    fh.close()

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

def generateIrisInterfaceCode(args, input):
    codes = []
    for file in input:
        print("Reading input: "+file)
        fh = open(file, 'r')
        lcodes = fh.readlines()
        for code in lcodes:
            codes.append(code)
        fh.close()
    #print("\n".join(codes))
    data = []
    data_hash = {}
    header_data_hash = {}
    for code_msg in codes:
        d_code = [parseFnParams(code_msg)]
        code = int(d_code[0][0])
        d = [d_code[0][1:]]
        if code == 0:
            kname = d[0][1]
            kname = re.sub(r'"', '', kname)
            kname = re.sub(r'\s*', '', kname)
            d[0][1] = kname
            data.append(d[0])
            data_hash[kname] = preprocess_data(d[0])
        elif code == 1:
            (core_api_name, task_api_name) = d[0][0], d[0][1]
            core_api_name = remove_spaces(core_api_name)
            task_api_name = remove_spaces(task_api_name)
            d = [['task0'] + d[0][2:]]
            kname = remove_spaces(d[0][1])
            d[0][1] = kname
            data.append(d[0])
            data_hash[kname] = preprocess_data(d[0])
            header_data_hash[core_api_name] = (0, data_hash[kname])
            header_data_hash[task_api_name] = (1, data_hash[kname])
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
    appendKernelSignatureHeaderFile(args, sig_lines, header_data_hash)
    #lines = lines + sig_lines
    k_sig_lines = [''' 
#ifdef __cplusplus
extern "C" {
#endif
    ''']
    k_sig_lines += sig_lines
    k_sig_lines.append('''
#ifdef __cplusplus
}
#endif
    ''')
    writeLinesToFiles(k_sig_lines, args.signature_file)
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
    wfh = open(args.output, 'w')
    print("Consolidated CPU/DSP interface code for all kernels in header file:"+args.output)
    wfh.write("\n".join(lines)+"\n")
    wfh.close()

def getKernelParamVairable(k):
    return "__k_"+k+"_args"

def appendConditionalParameters(expr, fdt, params, lines):
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
        flag = expr_eval & res[k]
        if not flag:
            continue
        if k == 'iris_dsp':
            lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            lines.append("#endif")
        elif k == 'iris_cpu':
            lines.append("#ifdef ENABLE_IRIS_OPENMP_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            lines.append("#endif")
        elif k == 'iris_nvidia':
            lines.append("#ifdef ENABLE_IRIS_HOST2OPENCL_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            lines.append("#endif")
            lines.append("#ifdef ENABLE_IRIS_HOST2CUDA_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            lines.append("#endif")
        elif k == 'iris_amd':
            lines.append("#ifdef ENABLE_IRIS_HOST2OPENCL_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            lines.append("#endif")
            lines.append("#ifdef ENABLE_IRIS_HOST2HIP_APIS")
            lines.append("\t\t\t\t"+fdt+",")
            lines.append("#endif")

def writeLinesToFiles(lines, filename):
    wfh = open(filename, 'w')
    print("Consolidated CPU/DSP interface code for all kernels in header file:"+args.output)
    wfh.write("\n".join(lines)+"\n")
    wfh.close()

def appendKernelSignatureHeaderFile(args, lines, data_hash):
    global valid_params, global_res
    def InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines):
        params.append(fdt+" "+fvar)
    for hdr in args.signature_file_include_headers:
        hdr_list = hdr.split(",")
        for each_hdr in hdr_list:
            each_hdr_str = remove_spaces(each_hdr)
            lines.append(f"#include \"{each_hdr_str}\"")
    for k,(hdr_type, v) in data_hash.items():
        kvar = getKernelParamVairable(k)
        func_sig = "int "+k+"("
        params = []
        lines.append(func_sig)
        arguments_start_index = find_start_index(v)
        i = arguments_start_index
        while (i<len(v)):
            f_param = v[i]
            i = i+1
            if f_param not in valid_params:
                continue
            f_details = v[i]
            i = i+1
            fvar = f_details[0]
            fdt = f_details[1]
            if f_param == 'PARAM' and len(f_details)>2:
                expr = getPyExprString(f_details[2])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'PARAM_CONST' and len(f_details)>3:
                expr = getPyExprString(f_details[2])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'VEC_PARAM' and len(f_details)>2:
                fdt  = fdt + f" * "
                expr = getPyExprString(f_details[2])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'IN_TASK':
                if hdr_type == 1:
                    fdt  = 'iris_mem' 
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'IN_TASK_OFFSET':
                if hdr_type == 1:
                    fdt  = 'iris_mem' 
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'OUT_TASK':
                if hdr_type == 1:
                    fdt  = 'iris_mem' 
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'OUT_TASK_OFFSET':
                if hdr_type == 1:
                    fdt  = 'iris_mem' 
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'IN_OUT_TASK':
                if hdr_type == 1:
                    fdt  = 'iris_mem' 
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            elif f_param == 'IN_OUT_TASK_OFFSET':
                if hdr_type == 1:
                    fdt  = 'iris_mem' 
                expr = getPyExprString(f_details[-1])
                InsertConditionalDeclarationsWithVariable(k, expr, fdt, fvar, params, lines)
                one_param_exists = True
            else:
                params.append(fdt+f" "+" "+fvar)
                one_param_exists = True
        if len(params) > 0:
             lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
        lines.append(");")

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
            f_param = v[i]
            i = i+1
            if f_param not in valid_params:
                continue
            f_details = v[i]
            i = i+1
            fvar = f_details[0]
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
    re_pattern = re.compile(r'^\s*IN_TASK|^\s*IN_OUT_TASK|^\s*OUT_TASK|^\s*PARAM|^\s*VEC_PARAM')
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
            f_param = v[i]
            i = i+1
            if f_param not in valid_params:
                continue
            f_details = v[i]
            i = i+1
            fvar = f_details[0]
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
        f_param = data[i]
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
        f_param = data[i]
        i = i+1
        if f_param not in valid_params:
            continue
        f_details = data[i]
        i = i+1
        fvar = f_details[0]
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

    def InsertConditionalParameters(kvar, k, expr, f_details, params, lines):
        appendConditionalParameters(expr, kvar+f_details[0], params, lines)

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
            params = []
            arguments_start_index = find_start_index(v)
            i = arguments_start_index
            one_param_exists = False
            while (i<len(v)):
                f_param = v[i]
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
                else:
                    params.append(kvar+f_details[0])
                    one_param_exists = True
                i = i+1
            if len(params) > 0:
                lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
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

def InitParser(parser):
    parser.add_argument("-input", dest="input", nargs='+', default="sample.c")
    parser.add_argument("-output", dest="output", default="sample.c.iris")
    parser.add_argument("-extract", dest="extract", action="store_true")
    parser.add_argument("-generate", dest="generate", action="store_true")
    parser.add_argument("-thread_safe", dest="thread_safe", type=int, default=1)
    parser.add_argument("-compile_options", dest="compile_options", default="")
    parser.add_argument("-signature_file", dest="signature_file", default="app.h")
    parser.add_argument("-signature_file_include_headers", dest="signature_file_include_headers", nargs='+', default=[])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    InitParser(parser)
    args = parser.parse_args()
    input = args.input
    if args.extract and args.generate:
        input = [args.output]
    if not args.extract and not args.generate:
        args.extract = True 
    if args.extract:
        iris_calls = extractTaskCalls(args)
        writeExtracts(iris_calls, args)
    if args.generate:
        generateIrisInterfaceCode(args, input)
