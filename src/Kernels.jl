import Base: +
import Base: -
import Base: /
import Base: *
function __iris_jacc_add(i, Z, X, Y)
    @inbounds Z[i] = X[i] + Y[i]
end
function __iris_jacc_sub(i, Z, X, Y)
    @inbounds Z[i] = X[i] - Y[i]
end
function __iris_jacc_mul(i, Z, X, Y)
    @inbounds Z[i] = X[i] * Y[i]
end
function __iris_jacc_div(i, Z, X, Y)
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
function __arith_kernel_v_s(kernel, a::IrisHRT.IrisMem, b; task=nothing, submit=true, wait=true, args...)
    kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_v_s")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = IrisHRT.dmem(jtype, sizes)
    Main.__iris_dmem_map[out.uid] = out

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    task = IrisHRT.parallel_for(Int64(total_size), kernel_fn, out, a, b; in=[a], out=[out], task=task, submit=submit, wait=wait, args...)
    return IrisHRT.IrisTaskMem(task, out)
end
function __arith_kernel_s_v(kernel, a, b::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, args...)
    kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_s_v")
    sizes = IrisHRT.host_size(b)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(b)
    type_str = IrisHRT.iris_get_type_string(b)

    # Create DMEM
    out = IrisHRT.dmem(jtype, sizes)
    Main.__iris_dmem_map[out.uid] = out

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    task = IrisHRT.parallel_for(Int64(total_size), kernel_fn, out, a, b; in=[b], out=[out], task=task, submit=submit, wait=wait, args...)
    return IrisHRT.IrisTaskMem(task, out)
end
function __arith_kernel(kernel, a::IrisHRT.IrisMem, b::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, args...)
    kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel)
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = IrisHRT.dmem(jtype, sizes)
    Main.__iris_dmem_map[out.uid] = out

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype, " wait:", wait, " submit:", submit, " task:", task)
    task = IrisHRT.parallel_for(Int64(total_size), kernel_fn, out, a, b; in=[a,b], out=[out], task=task, submit=submit, wait=wait, args...)
    return out
end
function __arith_kernel_v_s_out(kernel, out::IrisHRT.IrisMem, a::IrisHRT.IrisMem, b; task=nothing, submit=true, wait=true, args...)
    kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_v_s")
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    task = IrisHRT.parallel_for(Int64(total_size), kernel_fn, out, a, b; in=[a], out=[out], task=task, submit=submit, wait=wait, args...)
    return IrisHRT.IrisTaskMem(task, out)
end
function __arith_kernel_s_v_out(kernel, out::IrisHRT.IrisMem, a, b::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, args...)
    kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel * "_s_v")
    sizes = IrisHRT.host_size(b)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(b)
    type_str = IrisHRT.iris_get_type_string(b)

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    task = IrisHRT.parallel_for(Int64(total_size), kernel_fn, out, a, b; in=[b], out=[out], task=task, submit=submit, wait=wait, args...)
    return IrisHRT.IrisTaskMem(task, out)
end
function __arith_kernel_out(kernel, out::IrisHRT.IrisMem, a::IrisHRT.IrisMem, b::IrisHRT.IrisMem; task=nothing, submit=true, wait=true, args...)
    kernel_fn = IrisHRT.getFunction("__iris_jacc_" * kernel)
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create and submit JACC task
    if task == nothing
        task = IrisHRT.empty_task()
    end
    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype, " wait:", wait, " submit:", submit, " task:", task)
    task = IrisHRT.parallel_for(Int64(total_size), kernel_fn, out, a, b; in=[a,b], out=[out], task=task, submit=submit, wait=wait, args...)
    return out
end

######## Vector = Vector (Op) Vector
function +(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel("add", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end
function -(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel("sub", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end
function *(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel("mul", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end
function /(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel("div", IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end

######## Vector = Vector (Op) Scalar 
function +(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s("add", IrisHRT.field_dmem(a), b; args...)
end
function -(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s("sub", IrisHRT.field_dmem(a), b; args...)
end
function *(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s("mul", IrisHRT.field_dmem(a), b; args...)
end
function /(a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s("div", IrisHRT.field_dmem(a), b; args...)
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
function +(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel_out("add", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end
function -(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel_out("sub", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end
function *(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel_out("mul", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end
function /(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}; args...)
    return __arith_kernel_out("div", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), IrisHRT.field_dmem(b); args...)
end

######## Vector = Vector (Op) Scalar 
function +(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s_out("add", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), b; args...)
end
function -(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s_out("sub", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), b; args...)
end
function *(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s_out("mul", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), b; args...)
end
function /(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, b::T; args...) where T <: Number
    return __arith_kernel_v_s_out("div", IrisHRT.field_dmem(out), IrisHRT.field_dmem(a), b; args...)
end

######## Vector = Scalar (Op) Vector 
function +(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v_out("add", IrisHRT.field_dmem(out), a, field_mem(b))
end
function -(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v_out("sub", IrisHRT.field_dmem(out), a, field_mem(b))
end
function *(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v_out("mul", IrisHRT.field_dmem(out), a, field_mem(b))
end
function /(out::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}, a::T, b::Union{IrisHRT.IrisMem,IrisHRT.IrisTaskMem}) where T <: Number
    return __arith_kernel_s_v_out("div", IrisHRT.field_dmem(out), a, field_mem(b))
end

