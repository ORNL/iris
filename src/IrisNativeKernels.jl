

@inline function iris_task_cosf(task::UTask, out::UMem, in0::UMem, size::USize)
    return ccall(Libdl.dlsym(lib, :iris_task_cosf), Int32, (IrisTask, IrisMem, IrisMem, Csize_t), field_task(task), field_dmem(out), field_dmem(in0), Csize_t(size))
end

@inline function native_cosf(args...)
    return iris_task_cosf(args...)
end

@inline function iris_task_expf(task::UTask, out::UMem, in0::UMem, size::USize)
    return ccall(Libdl.dlsym(lib, :iris_task_expf), Int32, (IrisTask, IrisMem, IrisMem, Csize_t), field_task(task), field_dmem(out), field_dmem(in0), Csize_t(size))
end
@inline function native_expf(args...)
    return iris_task_expf(args...)
end

@inline function iris_task_expd(task::UTask, out::UMem, in0::UMem, size::USize)
    return ccall(Libdl.dlsym(lib, :iris_task_expd), Int32, (IrisTask, IrisMem, IrisMem, Csize_t), field_task(task), field_dmem(out), field_dmem(in0), Csize_t(size))
end
@inline function native_expd(args...)
    return iris_task_expd(args...)
end


@inline function iris_task_v_u_diff_expf(task::UTask, out::UMem, in0::UMem, in1::UMem, size::USize)
    return ccall(Libdl.dlsym(lib, :iris_task_v_u_diff_expf), Int32, (IrisTask, IrisMem, IrisMem, IrisMem, Csize_t), field_task(task), field_dmem(out), field_dmem(in0), field_dmem(in1), Csize_t(size))
end
@inline function native_v_u_diff_expf(args...)
    return iris_task_v_u_diff_expf(args...)
end

@inline function iris_task_rope_normalf(task::UTask, inout::UMem, rows::USize, cols::USize, pos::Number, theta_scale)
    return ccall(Libdl.dlsym(lib, :iris_task_rope_normalf), Int32, (IrisTask, IrisMem, Csize_t, Csize_t, Int32, Float32), field_task(task), field_dmem(inout), Csize_t(rows), Csize_t(cols), Int32(pos), Float32(theta_scale))
end
@inline function native_rope_normalf(args...)
    return iris_task_rope_normalf(args...)
end


