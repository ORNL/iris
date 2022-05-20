import re
import argparse
import pdb
import regex
import os
import tempfile

arguments_start_index=7
valid_params = { 'PARAM' : 1, 'PARAM_CONST' : 1, 'IN_TASK' : 1, 'IN_TASK_OFFSET' : 1, 'OUT_TASK_OFFSET' : 1, 'OUT_TASK' : 1, 'IN_OUT_TASK' : 1, 'IN_OUT_TASK_OFFSET' : 1, 'VEC_PARAM' : 1 }

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
    print(f"[Cmd] cpp {file} -Uiris_cpu -Uiris_dsp -o {f.name} {args.compile_options} -DUNDEF_IRIS_MACROS")
    os.system(f"cpp {file} -Uiris_cpu -Uiris_dsp -o {f.name} {args.compile_options} -DUNDEF_IRIS_MACROS")
    fh = open(f.name, "r")
    lines = fh.readlines()
    single_line = " ".join(lines);
    single_line = re.sub(r'\r', '', single_line)
    single_line = re.sub(r'\n', '', single_line)
    result = regex.findall(r'''IRIS_SINGLE_TASK
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
    fh.close()
    os.unlink(f.name)
    return result

def writeExtracts(code, args):
    fh = open(args.output, "w")
    for l in code:
        fh.write(l+"\n")
    fh.close()

def parse_nested(text, left=r'[(]', right=r'[)]', sep=r','):
    """ Based on https://stackoverflow.com/a/17141899/190597 (falsetru) """
    text = re.sub(r'\s', '', text)
    nstack = parseFnParams(text)
    pat = r'({}|{}|{})'.format(left, right, sep)
    tokens = re.split(pat, text)    
    tokens = [ x for x in tokens if x != '']
    stack = [[]]
    pdb.set_trace()
    for x in tokens:
        if not x or re.match(sep, x): continue
        if re.match(left, x):
            pdb.set_trace()
            stack[-1].append([])
            stack.append(stack[-1][-1])
        elif re.match(right, x):
            pdb.set_trace()
            stack.pop()
            if not stack:
                raise ValueError('error: opening bracket is missing')
        else:
            pdb.set_trace()
            stack[-1].append(x)
    if len(stack) > 1:
        print(stack)
        raise ValueError('error: closing bracket is missing')
    return stack.pop()

def parseFnParams(text):
    fn_name, nstack = parseFnParamsCore(text)
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
    for code in codes:
        d = [parseFnParams(code)]
        kname = d[0][1]
        kname = re.sub(r'"', '', kname)
        kname = re.sub(r'\s*', '', kname)
        d[0][1] = kname
        data.append(d[0])
        data_hash[kname] = preprocess_data(d[0])
    k_hash = {}
    k_index = 0
    for k,v in data_hash.items():
        k_hash[k] = k_index
        k_index = k_index+1
    lines = []
    lines.append('''
#ifndef __IRIS_CPU_DSP_INTERFACE_H__
#define __IRIS_CPU_DSP_INTERFACE_H__

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

static int iris_kernel_idx = 0;

        ''')
    lines.append('''
#ifdef ENABLE_IRIS_HEXAGON_APIS
#define iris_kernel_lock   iris_hexagon_lock
#define iris_kernel_unlock iris_hexagon_unlock
#define iris_kernel iris_hexagon_kernel
#define iris_setarg iris_hexagon_setarg
#define iris_setmem iris_hexagon_setmem
#define iris_launch iris_hexagon_launch
            ''')
    for k,v in data_hash.items():
        lines.append("#define iris_kernel_"+k+" irishxg_"+k)
    lines.append('''
#define HANDLE irishxg_handle_stub(), 
#define HANDLETYPE uint64_t,
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
#define HANDLE
#define HANDLETYPE 
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
#define HANDLE  iris_host2opencl_get_handle(), 
#define HANDLETYPE void *,
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
#define HANDLE 
#define HANDLETYPE 
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
#define HANDLE 
#define HANDLETYPE 
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
    appendStructure(lines, data_hash, k_hash)
    appendKernelSignature(lines, data_hash, k_hash)
    for k,v in data_hash.items():
        appendKernelSetArgMemParamFunctions(lines, k, v)
    appendKernelSetArgMemGlobalFunctions(lines, data_hash, k_hash)
    appendKernelLaunchFunction(lines, data_hash, k_hash)
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

def appendKernelSignature(lines, data_hash, k_hash):
    global valid_params
    def InsertConditionalDeclarations(k, expr, fdt, params, lines):
        res= { "iris_all" : 0xFFFFFFFF, 
            "iris_cpu"    : 0x01, 
            "iris_dsp"    : 0x02, 
            "iris_fpga"   : 0x04, 
            "iris_xfpga"  : 0x04, 
            "iris_ifpga"  : 0x08, 
            "iris_gpu"    : 0x0C,
        }
        if not re.match(r'.*iris_', expr):
            params.append(fdt)
            return
        expr = re.sub('\|\|', ' | ', expr)
        expr = re.sub('\&\&', ' & ', expr)
        #print("EXPR: "+expr)
        expr_eval = eval(expr, res)
        if (expr_eval & res['iris_cpu']) and (expr_eval & res['iris_dsp']):
           params.append(fdt)
        elif expr_eval & res['iris_dsp']:
           if len(params) > 0:
                lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
           lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
           lines.append("\t\t\t\t"+fdt+",")
           lines.append("#endif")
           params.clear()
        elif expr_eval & res['iris_cpu']:
           if len(params) > 0:
                lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
           lines.append("#ifndef ENABLE_IRIS_HEXAGON_APIS")
           lines.append("\t\t\t\t"+fdt+",")
           lines.append("#endif")
           params.clear()

    for k,v in data_hash.items():
        kvar = getKernelParamVairable(k)
        func_sig = "int iris_kernel_"+k+"(HANDLETYPE"
        params = []
        lines.append(func_sig)
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

def appendStructure(lines, data_hash, k_hash):
    global valid_params
    for k,v in data_hash.items():
        kvar = getKernelParamVairable(k)
        lines.append("typedef struct {")
        params = []
        param_hash = {}
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
        lines.append("static iris_"+kvar+" "+kvar+";")

def appendKernelSetArgMemParamFunctions(lines, kernel, data):
    global arguments_start_index
    global valid_params
    kvar = getKernelParamVairable(kernel)
    lines.append("static int iris_"+kernel+"_setarg(int idx, size_t size, void* value) {")
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
        if f_param=="PARAM":
            lines.append("\t\t\tcase "+str(p_index)+": memcpy(&"+kvar+"."+fvar+", value, size); break;")
        elif f_param=="PARAM_CONST":
            lines.append("\t\t\tcase "+str(p_index)+": memcpy(&"+kvar+"."+fvar+", value, size); break;")
        p_index = p_index+1
    lines.append('''
        default: return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}
        ''')
    lines.append("static int iris_"+kernel+"_setmem(int idx, void *mem, int size) {")
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
            lines.append("\t\t\tcase "+str(p_index)+": "+kvar+"."+fvar+" = ("+fdt+")mem; "+kvar+"."+fvar+"___size = size; break;")
        elif f_param=="VEC_PARAM":
            lines.append("\t\t\tcase "+str(p_index)+": "+kvar+"."+fvar+" = ("+fdt+"*)mem; break;")
        p_index = p_index+1
    lines.append('''
        default: return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}
        ''')

def appendKernelSetArgMemGlobalFunctions(lines, data_hash, k_hash):
    lines.append('''
int iris_setarg(int idx, size_t size, void* value) {
  switch (iris_kernel_idx) {
    ''')
    for k,v in data_hash.items():
        lines.append("\t\t\t case "+str(k_hash[k])+": return iris_"+k+"_setarg(idx, size, value);")
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
        lines.append("\t\t\t case "+str(k_hash[k])+": return iris_"+k+"_setmem(idx, mem, size);")
    lines.append('''
    }
    return IRIS_ERROR;
}
        ''')

def appendKernelLaunchFunction(lines, data_hash, k_hash):
    global arguments_start_index
    global valid_params
    lines.append('''
int iris_kernel(const char* name) {
    iris_kernel_lock();
    ''')
    for k,v in data_hash.items():
        lines.append("\t if (strcmp(name, \""+k+"\") == 0) {")
        lines.append("\t\t iris_kernel_idx = "+str(k_hash[k])+";")
        lines.append("\t\t return IRIS_SUCCESS;")
        lines.append("\t }")
    lines.append('''
    return IRIS_ERROR;
}
        ''')
    lines.append('''
int iris_launch(
#ifdef IRIS_ENABLE_STREAMS
        StreamType  stream,
#endif
        int dim, size_t off, size_t ndr) {
        switch(iris_kernel_idx) {
        ''')
    def InsertConditionalParameters(k, expr, f_details, params, lines):
        res= { "iris_all" : 0xFFFFFFFF, 
            "iris_cpu"    : 0x01, 
            "iris_dsp"    : 0x02, 
            "iris_fpga"   : 0x04, 
            "iris_xfpga"  : 0x04, 
            "iris_ifpga"  : 0x08, 
            "iris_gpu"    : 0x0C,
        }
        if not re.match(r'.*iris_', expr):
            params.append(getKernelParamVairable(k)+"."+f_details[0])
            return
        expr = re.sub('\|\|', ' | ', expr)
        expr = re.sub('\&\&', ' & ', expr)
        #print("EXPR: "+expr)
        expr_eval = eval(expr, res)
        if (expr_eval & res['iris_cpu']) and (expr_eval & res['iris_dsp']):
           params.append(getKernelParamVairable(k)+"."+f_details[0])
        elif expr_eval & res['iris_dsp']:
           if len(params) > 0:
                lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
           lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
           lines.append("\t\t\t\t"+getKernelParamVairable(k)+"."+f_details[0]+",")
           lines.append("#endif")
           params.clear()
        elif expr_eval & res['iris_cpu']:
           if len(params) > 0:
                lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
           lines.append("#ifndef ENABLE_IRIS_HEXAGON_APIS")
           lines.append("\t\t\t\t"+getKernelParamVairable(k)+"."+f_details[0]+",")
           lines.append("#endif")
           params.clear()

    for k,v in data_hash.items():
        lines.append("\t\t\tcase "+str(k_hash[k])+": iris_kernel_"+k+"(HANDLE STREAM_VAR")
        params = []
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
                InsertConditionalParameters(k, expr, f_details, params, lines)
                one_param_exists = True
            elif f_param == 'PARAM_CONST' and len(f_details)>3:
                expr = getPyExprString(f_details[2])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                one_param_exists = True
            elif f_param == 'VEC_PARAM' and len(f_details)>2:
                expr = getPyExprString(f_details[2])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                one_param_exists = True
            elif f_param == 'IN_TASK':
                expr = getPyExprString(f_details[-1])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\t("+getKernelParamVairable(k)+"."+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'IN_TASK_OFFSET':
                expr = getPyExprString(f_details[-1])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\t("+getKernelParamVairable(k)+"."+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'OUT_TASK':
                expr = getPyExprString(f_details[-1])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\t("+getKernelParamVairable(k)+"."+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'OUT_TASK_OFFSET':
                expr = getPyExprString(f_details[-1])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\t("+getKernelParamVairable(k)+"."+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'IN_OUT_TASK':
                expr = getPyExprString(f_details[-1])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\t("+getKernelParamVairable(k)+"."+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            elif f_param == 'IN_OUT_TASK_OFFSET':
                expr = getPyExprString(f_details[-1])
                InsertConditionalParameters(k, expr, f_details, params, lines)
                if len(params) > 0:
                     lines.append("\t\t\t\t"+", \n\t\t\t\t".join(params)+",")
                lines.append("#ifdef ENABLE_IRIS_HEXAGON_APIS")
                lines.append("\t\t\t\t("+getKernelParamVairable(k)+"."+f_details[0]+"___size)/sizeof("+f_details[2]+"),")
                lines.append("#endif")
                params.clear()
                one_param_exists = True
            else:
                params.append(getKernelParamVairable(k)+"."+f_details[0])
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
        lines.append("\t\t\t\t);");
        lines.append("\t\t\t\t\t break;")
    lines.append('''
        }
        iris_kernel_unlock();
        return IRIS_SUCCESS;
}
        ''')

def InitParser(parser):
    parser.add_argument("-input", dest="input", nargs='+', default="sample.c")
    parser.add_argument("-output", dest="output", default="sample.c.iris")
    parser.add_argument("-extract", dest="extract", action="store_true")
    parser.add_argument("-generate", dest="generate", action="store_true")
    parser.add_argument("-compile_options", dest="compile_options", default="")

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
