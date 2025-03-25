# Test for conjugate gradients
# The couple of axpy and dot products can be computed in parallel.
# Also the last axpy and dot product can be run in parallel too.
# Only two GPUs can be used. Using large arrays can affect the error,
# so other things like  pre-conditioner can be necessary,
# but we can focus on this part only for the performance analysis,
# without considering these constraints.
include(ENV["IRIS"] * "/include/iris/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

function matvecmul(i, a1, a2, a3, x, y, SIZE)
    if i == 1
        y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
    elseif i == SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
    elseif i > 1 && i < SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i] + a1[i] * x[i + 1]
    end
end
                                                                             

function dot(i, x, y)
    @inbounds return x[i] * y[i]
end

function axpy(i, alpha, x, y)
    @inbounds x[i] += alpha[1, 1] * y[i]
end

include("dot_product1.jl")

SIZE = 10000000
#SIZE = 16
a0 = IrisHRT.ones(Float64, SIZE)
a1 = IrisHRT.fill(Float64, 4, SIZE)
a2 = IrisHRT.ones(Float64, SIZE)
r = IrisHRT.fill(Float64, 0.5, SIZE)
p = IrisHRT.fill(Float64, 0.5, SIZE)
s = IrisHRT.dmem(Float64, SIZE)
x = IrisHRT.zeros(Float64, SIZE)

global cond = one(Float64)

r_aux = IrisHRT.dmem(Float64, SIZE)
r_old = IrisHRT.dmem(Float64, SIZE)
julia_start = time()
function parallel_reduce(task, SIZE, f, x, y, z; eargs...)
    IrisHRT.task(in=[x,y], out=[z], gws=[SIZE], host=true,
            task=task, submit=submit, wait=wait,
            kernel="gpu_dot", 
            args=[x, y, z])
end

async_flag = true
submit = true
wait = !async_flag

beta = IrisHRT.dmem(Float64, 1)
alpha = IrisHRT.dmem(Float64, 1)
negative_alpha = IrisHRT.dmem(Float64, 1)

alpha0 = IrisHRT.zeros(Float64, 1)
alpha1 = IrisHRT.zeros(Float64, 1)
beta0  = IrisHRT.zeros(Float64, 1)
beta1  = IrisHRT.zeros(Float64, 1)
ccond  = IrisHRT.zeros(Float64, 1)

alpha0_buffer = IrisHRT.zeros(Float64, SIZE)
alpha1_buffer = IrisHRT.zeros(Float64, SIZE)
beta0_buffer = IrisHRT.zeros(Float64, SIZE)
beta1_buffer = IrisHRT.zeros(Float64, SIZE)
ccond_buffer = IrisHRT.zeros(Float64, SIZE)
iterations_max = 50
index = 0
while cond[1, 1] >= 1e-14 && index < iterations_max
    global p
	global index

    julia_inner_start = time()
    task = nothing 
	if !submit
		task = IrisHRT.empty_task()
	end
    IrisHRT.copyto!(r_old, r; task=task, submit=submit, wait=wait)
    IrisHRT.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE; task=task, submit=submit, wait=wait)

    IrisHRT.parallel_reduce(SIZE, dot, alpha0, alpha0_buffer, r, r; task=task, submit=submit, wait=wait) 
    IrisHRT.parallel_reduce(SIZE, dot, alpha1, alpha1_buffer, p, s; task=task, submit=submit, wait=wait)

    /(alpha, alpha0, alpha1; task=task, submit=submit, wait=wait)
    *(negative_alpha, alpha, -1.0; task=task, submit=submit, wait=wait)

    IrisHRT.parallel_for(SIZE, axpy, negative_alpha, r, s; task=task, submit=submit, wait=wait)
    IrisHRT.parallel_for(SIZE, axpy, alpha, x, p; task=task, submit=submit, wait=wait)
    IrisHRT.parallel_reduce(SIZE, dot, beta1, beta1_buffer, r_old, r_old; task=task, submit=submit, wait=wait)
    IrisHRT.parallel_reduce(SIZE, dot, beta0, beta0_buffer, r, r; task=task, submit=submit, wait=wait)
    /(beta, beta0, beta1; task=task, submit=submit, wait=wait)

    IrisHRT.copyto!(r_aux, r; task=task, submit=submit, wait=wait)

    IrisHRT.parallel_for(SIZE, axpy, beta, r_aux, p; task=task, submit=submit, wait=wait)
    IrisHRT.parallel_reduce(SIZE, dot, ccond, ccond_buffer, r, r; task=task, submit=submit, wait=wait)
    IrisHRT.copyto!(p, r_aux; task=task, submit=submit, wait=wait)
    if task != nothing
        IrisHRT.submit(task)
    elseif !wait
        IrisHRT.synchronize()
    end
    julia_inner_time0 = time() - julia_inner_start
    IrisHRT.flush(ccond)
    global cond = IrisHRT.host(ccond)
    println(Core.stdout, " Cond: ", ccond, " time:", julia_inner_time0)
	index = index + 1
end
julia_time0 = time() - julia_start
println("Julia time: ", julia_time0)
println("Test:", cond[1, 1] <= 1e-14)
IrisHRT.finalize()
