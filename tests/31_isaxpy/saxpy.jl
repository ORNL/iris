using Julius
using LinearAlgebra
using Base.Threads

# Initialize IRIS runtime
Julius.init()

# Problem size
SIZE = length(ARGS) == 0 ? Int32(8) : parse(Int32, ARGS[1])
A = Int32(10)

function saxpy_openmp(Z, X, Y, N, A)
    nthreads = Threads.nthreads()
    num_cpus = Sys.CPU_THREADS
    Julius.@jmaybethreads for i in 1 : length(X)
        Z[i] = A * X[i] + Y[i]
    end
end


# Input vectors
x = collect(Int32.(0:SIZE-1))
y = collect(Int32.(0:SIZE-1))
s = zeros(Int32, SIZE)

# Create task
task=Julius.@jtask flush:[s], in:[x, y], submit:false, gws:[SIZE], saxpy(s, x, y, SIZE, A)  

# Submit to FPGA device
Julius.submit(task, Julius.iris_cpu)

println("S = ", A, " * X + Y = ", s)

# Finalize runtime
Julius.finalize()
