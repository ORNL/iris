const VALUE_SIZE = 8
#and(a, b) = a && b
#or(a, b) = a || b
function flat_in_buffer(value::Any)::IRISValue
    value_buffer = Base.zeros(UInt8, VALUE_SIZE)
    if typeof(value) == Float32
        value_buffer[1:4] .= reinterpret(UInt8, [value])
    elseif typeof(value) == Float64
        value_buffer[1:8] .= reinterpret(UInt8, [value])
    elseif typeof(value) == Int64
        value_buffer[1:8] .= reinterpret(UInt8, [value])
    elseif typeof(value) == Int32
        value_buffer[1:4] .= reinterpret(UInt8, [value])
    elseif typeof(value) == Int16
        value_buffer[1:2] .= reinterpret(UInt8, [value])
    elseif typeof(value) == Int8
        value_buffer[1:1] .= reinterpret(UInt8, [value])
    elseif typeof(value) == UInt64
        value_buffer[1:8] .= reinterpret(UInt8, [value])
    elseif typeof(value) == UInt32
        value_buffer[1:4] .= reinterpret(UInt8, [value])
    elseif typeof(value) == UInt16
        value_buffer[1:2] .= reinterpret(UInt8, [value])
    elseif typeof(value) == UInt8
        value_buffer[1:1] .= reinterpret(UInt8, [value])
    elseif typeof(value) == Bool 
        value_buffer[1:1] .= reinterpret(UInt8, [value])
    else
        println(Core.stdout, "Unknown type of $value")
    end
    ivalue = IRISValue(Tuple(value_buffer))
    #println("Element: ", value, " buffer:", ivalue)
    return ivalue 
end

function get_iris_type(T, default=iris_unknown)
    element_type = default
    if T == Float32
        element_type = iris_float 
    elseif T == Float64
        element_type = iris_double
    elseif T == Int64
        element_type = iris_int64
    elseif T == Int32
        element_type = iris_int32
    elseif T == Int16
        element_type = iris_int16
    elseif T == Int8
        element_type = iris_int8
    elseif T == UInt64
        element_type = iris_uint64
    elseif T == UInt32
        element_type = iris_uint32
    elseif T == UInt16
        element_type = iris_uint16
    elseif T == UInt8
        element_type = iris_uint8
    elseif T == Char
        element_type = iris_char
    elseif T == Bool
        element_type = iris_bool
    elseif T <: SubArray
        #println(Core.stdout, " calling param: ", T.parameters[1])
        element_type = get_iris_type(T.parameters[1])
    elseif isstructtype(T) 
        element_type = iris_custom_type
    elseif default == iris_pointer
        println(Core.stdout, "[Error] get_iris_type type is not found")
        element_type = iris_unknown
    end
    #println(Core.stdout, " get_iris_type returning element type: ", element_type, " ifloat:", iris_float, " custom_type:", iris_custom_type, " type:", T)
    return element_type
end

function iris_get_type_string(mem)
    element_type = iris_get_mem_element_type(mem)
    if element_type == iris_float 
        return "float"
    elseif element_type == iris_double
        return "double"
    elseif element_type == iris_int64
        return "int64_t" 
    elseif element_type == iris_int32
        return "int32_t"
    elseif element_type == iris_int16
        return "int16_t"
    elseif element_type == iris_int8
        return "int8_t"
    elseif element_type == iris_uint64
        return "uint64_t"
    elseif element_type == iris_uint32
        return "uint32_t"
    elseif element_type == iris_uint16
        return "uint16_t"
    elseif element_type == iris_uint8
        return "uint8_t"
    elseif element_type == iris_char
        return "char"
    elseif element_type == iris_bool
        return "bool"
    else
        error("Unknown iris type handler: ", element_type)
    end
    return "__unknown"
end

function iris_get_julia_type(mem::IrisMem)
    element_type = iris_get_mem_element_type(mem)
    if element_type == iris_float 
        return Float32
    elseif element_type == iris_double
        return Float64
    elseif element_type == iris_int64
        return Int64
    elseif element_type == iris_int32
        return Int32
    elseif element_type == iris_int16
        return Int16
    elseif element_type == iris_int8
        return Int8
    elseif element_type == iris_uint64
        return UInt64
    elseif element_type == iris_uint32
        return UInt32
    elseif element_type == iris_uint16
        return UInt16
    elseif element_type == iris_uint8
        return UInt8
    elseif element_type == iris_char
        return Char
    elseif element_type == iris_bool
        return Bool
    elseif element_type == iris_custom_type
        return get_dmem_custom_type(mem)
    else
        error("Unknown iris type for julia type: ", element_type)
    end
    return Ptr{Cvoid}
end

function call_stack(tag="")
    println("Backtrace ", tag, ": ")
    for (i, bt) in enumerate(Base.stacktrace())
        println("[$i] ", bt)
    end
end

# function that inspects the stack trace and returns the immediate caller’s function and line number.
function get_caller_info(; 
        exclude_fns::AbstractSet{Symbol}=Set{Symbol}())
    st = stacktrace()
    # st[1] is get_caller_info itself; start from st[2].
    exclude_files = Set( 
        ["IrisHRT.jl", "IrisUtils.jl", "IrisNativeKernels.jl", "IrisJuliaKernels.jl"]
    )
    for frame in st[2:end]
        fn_sym = frame.func
        file_name = basename(String(frame.file))
        if !(fn_sym in exclude_fns) && !(file_name in exclude_files)
            return (
                func   = fn_sym,
                file   = frame.file,
                line   = frame.line
            )
        end
    end
    return nothing
end

function iris_print_array(data::Ptr{T}, N::Int) where T
    str = []
    arr = unsafe_wrap(Array, data, N)
    for i in 1:N
        value = string.(arr[i])
        #iris_println("Data Value: $value")
        push!(str, value)
    end
    str_data = "[" * join(str, ",") * "]"
    iris_println("Data: $str_data")
end

function vec_string(arr::AbstractArray, sep::String = ", ")
    return "[" * join(string.(arr), sep) * "]"
end

function get_jacc_init(op, out_mem)
    out_type = IrisHRT.get_julia_type(out_mem)
    init = zero(out_type)
    if op == +
        init = JACC.default_init(out_type, op)
    elseif op == -
        init = zero(out_type)
    elseif op == *
        init = one(out_type)
    elseif op == /
        init = one(out_type)
    elseif op == sum
        init = zero(out_type)
    elseif op == prod
        init = one(out_type)
    elseif op == max
        init = out_type(-Inf)
    elseif op == min
        init = out_type(Inf)
    end
    return init
end

function convert_jacc_op_2_iris_op(op)
    iris_op = IrisHRT.iris_jacc_op_plus
    if op == nothing
        iris_op = IrisHRT.iris_jacc_op_plus
    elseif op == +
        iris_op = IrisHRT.iris_jacc_op_plus
    elseif op == -
        iris_op = IrisHRT.iris_jacc_op_minus
    elseif op == *
        iris_op = IrisHRT.iris_jacc_op_mul
    elseif op == /
        iris_op = IrisHRT.iris_jacc_op_div
    elseif op == Main.and
        iris_op = IrisHRT.iris_jacc_op_and
    elseif op == Main.or
        iris_op = IrisHRT.iris_jacc_op_or
    elseif op == sum
        iris_op = IrisHRT.iris_jacc_op_sum
    elseif op == prod
        iris_op = IrisHRT.iris_jacc_op_prod
    elseif op == min 
        iris_op = IrisHRT.iris_jacc_op_min
    elseif op == max
        iris_op = IrisHRT.iris_jacc_op_max
    else
        error("parallel_reduce_core with unknown op:", op)
    end
    return iris_op
end

function get_jacc_op(iris_op)
    if iris_op == iris_jacc_op_plus 
        op = +
    elseif iris_op == iris_jacc_op_minus
        op = -
    elseif iris_op == iris_jacc_op_mul  
        op = *
    elseif iris_op == iris_jacc_op_div  
        op = /
    elseif iris_op == iris_jacc_op_and
        op = Main.and
    elseif iris_op == iris_jacc_op_or
        op = Main.or
    elseif iris_op == iris_jacc_op_sum  
        op = Base.sum
    elseif iris_op == iris_jacc_op_prod 
        op = Base.prod
    elseif iris_op == iris_jacc_op_min  
        op = Base.min 
    elseif iris_op == iris_jacc_op_max  
        op = Base.max
    else
        op = +
    end
    return op
end

function parse_function(s::AbstractString)
    parts = split(s, ":")
    if length(parts) == 3
        include_file, mod, func = parts
        if mod == ""
            mod = nothing
        end
    elseif length(parts) == 2
        # Assume missing include-file; first part is the module
        include_file = nothing
        mod, func = parts
    elseif length(parts) == 1
        include_file = nothing
        mod = nothing
        func = parts[1]
    else
        error("Unexpected format: too many dots")
    end
    return (include_file, mod, func)
end
function getFunction(func_name)
    (include_file, mod, func) = parse_function(func_name)
    func_s = nothing

    if include_file != nothing
        include(include_file * ".jl")
        if mod != nothing
            # Check if the module is defined in the current module
            if isdefined(@__MODULE__, Symbol(mod))
                mod_s = getfield(@__MODULE__, Symbol(mod))
                # Check if the function exists in the module
                if isdefined(mod_s, Symbol(func))
                    func_s = getfield(mod_s, Symbol(func))
                end
            end
        else
            # No module provided, so check in the current module
            if isdefined(@__MODULE__, Symbol(func))
                func_s = getfield(@__MODULE__, Symbol(func))
            end
        end
    elseif mod != nothing
        # Check if the module is defined in Main
        if isdefined(Main, Symbol(mod))
            mod_s = getfield(Main, Symbol(mod))
            # Check if the function exists in the module
            if isdefined(mod_s, Symbol(func))
                func_s = getfield(mod_s, Symbol(func))
            end
        end
    else
        # Check directly in Main for the function
        if isdefined(Main, Symbol(func))
            func_s = getfield(Main, Symbol(func))
        end
    end
    return func_s
end

function getFunction_old(func_name)
    (include_file, mod, func) = parse_function(func_name)
    func_s = nothing
    if include_file != nothing
        #println(Core.stdout, "Including file:", include_file)
        include(include_file * ".jl")
        if mod != nothing
            mod_s = getfield(@__MODULE__, Symbol(mod))
            func_s = getfield(mod_s, Symbol(func))
        else
            #println(Core.stdout, "Get symbol: ", func)
            func_s = getfield(@__MODULE__, Symbol(func))
            #println(Core.stdout, "Get symbol: ", func_s)
        end
    elseif mod != nothing
        mod_s = getfield(Main, Symbol(mod))
        func_s = getfield(mod_s, Symbol(func))
    else
        func_s = getfield(Main, Symbol(func))
    end
    return func_s
end


