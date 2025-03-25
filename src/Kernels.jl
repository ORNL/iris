import Base: +
import Base: -
import Base: /
import Base: *
function __iris_jacc_log(i, Y, X)
    @inbounds Y[i] = Base.log(X[i])
end
function __iris_jacc_log(i, Y, b::Int64, X)
    @inbounds Y[i] = Base.log(b, X[i])
end
function __iris_jacc_log(i, Y, B, X)
    @inbounds Y[i] = Base.log(B[i], X[i])
end
function __iris_jacc_ldexp(i, Y, X, n::Int64)
    @inbounds Y[i] = Base.ldexp(X[i], n)
end
function __iris_jacc_ldexp(i, Y, X, n::Int64, alpha)
    @inbounds Y[i] = alpha * Base.ldexp(X[i], n)
end
function __iris_jacc_ldexp(i, Y, X, N::AbstractArray{Int64})
    @inbounds Y[i] = Base.ldexp(X[i], N[i])
end
function __iris_jacc_ldexp(i, Y, X, N::AbstractArray{Int64}, alpha)
    @inbounds Y[i] = alpha * Base.ldexp(X[i], N[i])
end
function __iris_jacc_add_v_v(i, Z, X, Y)
    @inbounds Z[i] = X[i] + Y[i]
end
function __iris_jacc_sub_v_v(i, Z, X, Y)
    @inbounds Z[i] = X[i] - Y[i]
end
function __iris_jacc_mul_v_v(i, Z, X, Y)
    @inbounds Z[i] = X[i] * Y[i]
end
function __iris_jacc_div_v_v(i, Z, X, Y)
    @inbounds Z[i] = X[i] / Y[i]
end
function __iris_jacc_add_v_s(i, Z, X, Ys)
    @inbounds Z[i] = X[i] + Ys
end
function __iris_jacc_sub_v_s(i, Z, X, Ys)
    @inbounds Z[i] = X[i] - Ys
end
function __iris_jacc_mul_v_s(i, Z, X, Ys)
    @inbounds Z[i] = X[i] * Ys
end
function __iris_jacc_div_v_s(i, Z, X, Ys)
    @inbounds Z[i] = X[i] / Ys
end
function __iris_jacc_add_s_v(i, Z, Xs, Y)
    @inbounds Z[i] = Xs + Y[i]
end
function __iris_jacc_sub_s_v(i, Z, Xs, Y)
    @inbounds Z[i] = Xs - Y[i]
end
function __iris_jacc_mul_s_v(i, Z, Xs, Y)
    @inbounds Z[i] = Xs * Y[i]
end
function __iris_jacc_div_s_v(i, Z, Xs, Y)
    @inbounds Z[i] = Xs / Y[i]
end
function __arith_kernel_v(kernel, a::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_v"
    #kernel_fn = IrisHRT.getFunction(kernel_name)
    #println(Core.stdout, "F call: ", kernel_name, " f:", kernel_fn)
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, a; in=[a], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[a], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, a],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end
function __arith_kernel_v_s(kernel, a::IrisHRT.IrisMem, b; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_v_s"
    #kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_v_s")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, a, b; in=[a], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[a], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, a, b],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end
function __arith_kernel_v_s_s(kernel, a::IrisHRT.IrisMem, scalar1, scalar2; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_v_s_s"
    #kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_v_s_s")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, a, scalar1, scalar2; in=[a], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[a], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, a, scalar1, scalar2],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end
function __arith_kernel_v_v(kernel, a::IrisHRT.IrisMem, b::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_v_v"
    #kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_v_v")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype, " wait:", wait, " submit:", submit, " task:", task)
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, a, b; in=[a,b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[a, b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, a, b],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end
function __arith_kernel_v_v_s(kernel, a::IrisHRT.IrisMem, b::IrisHRT.IrisMem, scalar; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_v_v_s"
    #kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_v_v_s")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype, " wait:", wait, " submit:", submit, " task:", task)
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, a, b, scalar; in=[a,b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[a, b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, a, b, scalar],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end
function __arith_kernel_v_v_s_s(kernel, a::IrisHRT.IrisMem, b::IrisHRT.IrisMem, scalar1, scalar2; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_v_v_s_s"
    #kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel *"_v_v_s_s")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype, " wait:", wait, " submit:", submit, " task:", task)
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, a, b, scalar1, scalar2; in=[a,b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[a, b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, a, b, scalar1, scalar2],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end
function __arith_kernel_s_v(kernel, scalar, b::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, flush=true, jacc=true, result=nothing, eargs...)
    kernel_name = "__iris_jacc_" * kernel * "_s_v"
    #kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_s_v")
    sizes = IrisHRT.host_size(b)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(b)
    type_str = IrisHRT.iris_get_type_string(b)

    # Create DMEM
    out = result
    if out == nothing
        out = IrisHRT.dmem(jtype, sizes)
        Main.__iris_dmem_map[out.uid] = out
    end

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    out_m, flush_m = [], []
    if flush
        push!(flush_m, out)
    else
        push!(out_m, out)
    end
    if jacc
        task = IrisHRT.parallel_for(Int64(total_size), kernel_name, out, scalar, b; in=[b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, eargs...)
    else
        task = IrisHRT.task(in=[b], out=out_m, flush=flush_m, task=task, submit=submit, wait=wait, 
                kernel=kernel_name,
                args=[Int64(total_size), out, scalar, b],
                eargs...)
    end
    if result == nothing
        return IrisHRT.IrisTaskMem(task, out)
    else
        return task
    end
end

######## Vector = Vector (Op) Vector
function +(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("add", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); eargs...)
end
function -(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("sub", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); eargs...)
end
function *(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("mul", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); eargs...)
end
function /(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("div", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); eargs...)
end

######## Vector = Vector (Op) Scalar 
function +(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("add", IrisHRT.field_dmem(a), b; eargs...)
end
function -(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("sub", IrisHRT.field_dmem(a), b; eargs...)
end
function *(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("mul", IrisHRT.field_dmem(a), b; eargs...)
end
function /(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("div", IrisHRT.field_dmem(a), b; eargs...)
end

######## Vector = Scalar (Op) Vector 
function +(a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("add", a, field_mem(b))
end
function -(a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("sub", a, field_mem(b))
end
function *(a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("mul", a, field_mem(b))
end
function /(a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("div", a, field_mem(b))
end

###################################
######## Vector = Vector (Op) Vector
function +(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("add", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); result=IrisHRT.field_dmem(out), eargs...)
end
function -(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("sub", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); result=IrisHRT.field_dmem(out), eargs...)
end
function *(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("mul", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); result=IrisHRT.field_dmem(out), eargs...)
end
function /(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("div", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); result=IrisHRT.field_dmem(out), eargs...)
end

######## Vector = Vector (Op) Scalar 
function +(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("add", IrisHRT.field_dmem(a), b; result=IrisHRT.field_dmem(out), eargs...)
end
function -(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("sub", IrisHRT.field_dmem(a), b; result=IrisHRT.field_dmem(out), eargs...)
end
function *(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("mul", IrisHRT.field_dmem(a), b; result=IrisHRT.field_dmem(out), eargs...)
end
function /(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; eargs...) where T <: Number
    return __arith_kernel_v_s("div", IrisHRT.field_dmem(a), b; result=IrisHRT.field_dmem(out), eargs...)
end

######## Vector = Scalar (Op) Vector 
function +(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("add", a, field_mem(b); result=IrisHRT.field_dmem(out), )
end
function -(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("sub", a, field_mem(b); result=IrisHRT.field_dmem(out), )
end
function *(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("mul", a, field_mem(b); result=IrisHRT.field_dmem(out), )
end
function /(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v("div", a, field_mem(b); result=IrisHRT.field_dmem(out), )
end

################### Define Common macros to define math functions #################
macro __core_math_define(fn_names...)
    exprs = []
    for name in fn_names
        # Escaping the name ensures the defined function is in the caller's scope.
        name_str = string(name) #string(QuoteNode(name))
        kname = string(name)
        system_func = name
        #println(Core.stdout, " name_str:", name_str, " kname:", kname, " system_func:", system_func)
        jacc_func_v = Symbol("__iris_jacc_" * kname * "_v")
        jacc_func_v_v = Symbol("__iris_jacc_" * kname * "_v_v")
        jacc_func_v_s = Symbol("__iris_jacc_" * kname * "_v_s")
        jacc_func_v_s_s = Symbol("__iris_jacc_" * kname * "_v_s_s")
        jacc_func_v_v_s = Symbol("__iris_jacc_" * kname * "_v_v_s")
        jacc_func_v_v_s_s = Symbol("__iris_jacc_" * kname * "_v_v_s_s")
        push!(exprs, quote
            function $(esc(jacc_func_v))(i, Y, X)
                @inbounds Y[i] = Base.$system_func(X[i])
            end
            function $(esc(jacc_func_v_s))(i, Y, X, alpha)
                @inbounds Y[i] = alpha * Base.$(esc(system_func))(X[i])
            end
            function $(esc(jacc_func_v_s_s))(i, Y, X, alpha, beta)
                @inbounds Y[i] = beta + alpha * Base.$(esc(system_func))(X[i])
            end
            function $(esc(jacc_func_v_v))(i, Y, X, B)
                @inbounds Y[i] = B[i] + Base.$(esc(system_func))(X[i])
            end
            function $(esc(jacc_func_v_v_s))(i, Y, X, B, alpha)
                @inbounds Y[i] = B[i] + alpha * Base.$(esc(system_func))(X[i])
            end
            function $(esc(jacc_func_v_v_s_s))(i, Y, X, B, alpha, beta)
                @inbounds Y[i] = beta * B[i] + alpha * Base.$(esc(system_func))(X[i])
            end
            #  C = op(A)
            function $(esc(name))(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
                return __arith_kernel_v($name_str, IrisHRT.field_dmem(a); eargs...)
            end
            #  C = alpha * op(A) 
            function $(esc(name))(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T; eargs...) where T <: Number
                return __arith_kernel_v_s($name_str, IrisHRT.field_dmem(a), alpha; eargs...)
            end
            #  C = beta + alpha * op(A) 
            function $(esc(name))(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T, beta::T; eargs...) where T <: Number
                return __arith_kernel_v_s_s($name_str, IrisHRT.field_dmem(a), alpha, beta; eargs...)
            end
            #  C = B + op(A)
            function $(esc(name))(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...) 
                return __arith_kernel_v_v($name_str, IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); eargs...)
            end
            #  C = B + alpha * op(A)
            function $(esc(name))(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T; eargs...) where T <: Number
                return __arith_kernel_v_v_s($name_str, IrisHRT.field_dmem(a), IrisHRT.field_dmem(b), alpha; eargs...)
            end
            #  C = beta * B + alpha * op(A)
            function $(esc(name))(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T, beta::T; eargs...) where T <: Number
                return __arith_kernel_v_v_s_s($name_str, IrisHRT.field_dmem(a), IrisHRT.field_dmem(b), alpha, beta; eargs...)
            end
            #  op_wrapper(C, op(A))
            function $(esc(name))(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
                return __arith_kernel_v($name_str, IrisHRT.field_dmem(a); result=IrisHRT.field_dmem(out), eargs...)
            end
            #  op_wrapper(C, alpha * op(A))
            function $(esc(name))(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T; eargs...) where T <: Number
                return __arith_kernel_v_s($name_str, IrisHRT.field_dmem(a), alpha; result=IrisHRT.field_dmem(out), eargs...)
            end
            #  op_wrapper(C, beta + alpha * op(A))
            function $(esc(name))(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T, beta::T; eargs...) where T <: Number
                return __arith_kernel_v_s_s($name_str, IrisHRT.field_dmem(a), alpha, beta; result=IrisHRT.field_dmem(out), eargs...)
            end
            #  op_wrapper(C, B + op(A))
            function $(esc(name))(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...) 
                return __arith_kernel_v_v($name_str, IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); result=IrisHRT.field_dmem(out), eargs...)
            end
            #  op_wrapper(C, B + alpha * op(A))
            function $(esc(name))(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T; eargs...) where T <: Number
                return __arith_kernel_v_v_s($name_str, IrisHRT.field_dmem(a), IrisHRT.field_dmem(b), alpha; result=IrisHRT.field_dmem(out), eargs...)
            end
            #  op_wrapper(C, beta * B + alpha * op(A))
            function $(esc(name))(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T, beta::T; eargs...) where T <: Number
                return __arith_kernel_v_v_s_s($name_str, IrisHRT.field_dmem(a), IrisHRT.field_dmem(b), alpha, beta; result=IrisHRT.field_dmem(out), eargs...)
            end
        end)
    end
    return Expr(:block, exprs...)
end

################### Define math functions #################
@__core_math_define sin asin sinh asinh sinpi asinpi
@__core_math_define cos acos cosh acosh cospi acospi
@__core_math_define tan atan tanh atanh tanpi atanpi     
@__core_math_define exp exp2 exp10                
@__core_math_define log2 log10 logb ilogb log1p     # Deal log separately
@__core_math_define sqrt cbrt rsqrt
#@__core_math_define floor ceil trunc

######### ldexp ############
function ldexp(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, n::Int64; eargs...)
    return __arith_kernel_v_s("ldexp", IrisHRT.field_dmem(a), n; eargs...)
end
function ldexp(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, n::Int64, alpha::T; eargs...) where T <: Number
    return __arith_kernel_v_s_s("ldexp", IrisHRT.field_dmem(a), n, alpha; eargs...)
end
function ldexp(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, n::Int64; eargs...)
    return __arith_kernel_v_s("ldexp", IrisHRT.field_dmem(a), n; result=IrisHRT.field_dmem(out), eargs...)
end
function ldexp(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, n::Int64, alpha; eargs...)
    return __arith_kernel_v_s_s("ldexp", IrisHRT.field_dmem(a), n, alpha; result=IrisHRT.field_dmem(out), eargs...)
end

function ldexp(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("ldexp", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); eargs...)
end
function ldexp(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha::T; eargs...) where T <: Number
    return __arith_kernel_v_v_s("ldexp", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b), alpha; eargs...)
end
function ldexp(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("ldexp", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); result=IrisHRT.field_dmem(out), eargs...)
end
function ldexp(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, alpha; eargs...)
    return __arith_kernel_v_v_s("ldexp", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b), alpha; result=IrisHRT.field_dmem(out), eargs...)
end

######### log ############
function log(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v("log", IrisHRT.field_dmem(a); eargs...)
end
function log(b::Int64, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_s_v("log", b, IrisHRT.field_dmem(a); eargs...)
end
function log(b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("log", IrisHRT.field_dmem(b), IrisHRT.field_dmem(a); eargs...)
end

function log(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v("log", IrisHRT.field_dmem(a); result=IrisHRT.field_dmem(out), eargs...)
end
function log(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Int64, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_s_v("log", b, IrisHRT.field_dmem(a); result=IrisHRT.field_dmem(out), eargs...)
end
function log(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; eargs...)
    return __arith_kernel_v_v("log", IrisHRT.field_dmem(b), IrisHRT.field_dmem(a); result=IrisHRT.field_dmem(out), eargs...)
end


