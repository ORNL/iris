import Base: +
import Base: -
import Base: /
import Base: *
function __iris_jacc_add(i, Z, X, Y)
    if i <= length(Z)
        @inbounds Z[i] = X[i] + Y[i]
    end
end
function __iris_jacc_sub(i, Z, X, Y)
    if i <= length(Z)
        @inbounds Z[i] = X[i] - Y[i]
    end
end
function __iris_jacc_mul(i, Z, X, Y)
    if i <= length(Z)
        @inbounds Z[i] = X[i] * Y[i]
    end
end
function __iris_jacc_div(i, Z, X, Y)
    if i <= length(Z)
        @inbounds Z[i] = X[i] / Y[i]
    end
end
function __arith_kernel(kernel, a::IrisHRT.IrisMem, b::IrisHRT.IrisMem)
    sizes = IrisHRT.host_size(a)
    total_size = prod(sizes)
    jtype = IrisHRT.get_julia_type(a)
    type_str = IrisHRT.iris_get_type_string(a)

    # Create DMEM
    out = IrisHRT.dmem(jtype, sizes)
    Main.__iris_dmem_map[out.uid] = out

    #println(Core.stdout, "sizes: ", sizes, " total_size:", total_size, " type:", jtype)

    # Create and submit JACC task
    task = IrisHRT.empty_task()
    task = IrisHRT.parallel_for(Int64(total_size), __iris_jacc_add, out, a, b, in=[a,b], out=[out], task=task, submit=true, wait=false)
    return out
end
function +(a::IrisHRT.IrisMem, b::IrisHRT.IrisMem)
    return __arith_kernel("add", a, b)
end
function -(a::IrisHRT.IrisMem, b::IrisHRT.IrisMem)
    return __arith_kernel("sub", a, b)
end
function *(a::IrisHRT.IrisMem, b::IrisHRT.IrisMem)
    return __arith_kernel("mul", a, b)
end
function /(a::IrisHRT.IrisMem, b::IrisHRT.IrisMem)
    return __arith_kernel("div", a, b)
end
