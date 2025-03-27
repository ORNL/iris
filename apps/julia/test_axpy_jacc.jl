include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

function axpy(i, alpha, x, y)
  @inbounds x[i] += alpha * y[i]
end

N = 10
# Generate random vectors x and y of length N for the interval [0, 100]
x = round.(rand(Float32, N) * 100)
y = round.(rand(Float32, N) * 100)
alpha = 2.5

x_d = IrisHRT.array(x)
y_d = IrisHRT.array(y)
IrisHRT.parallel_for(N, axpy, alpha, x_d, y_d)
IrisHRT.synchronize()

IrisHRT.finalize()
