include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

function axpy(i, alpha, x, y)
  @inbounds x[i] += alpha * y[i]
end

function axpy_test(alpha, x, y)
  for i in 1:length(x)
    x[i] += alpha * y[i]
  end
end

N = 10
# Generate random vectors x and y of length N for the interval [0, 100]
x = round.(rand(Float32, N) * 100)
y = round.(rand(Float32, N) * 100)
alpha = 2.5
x_ref = copy(x)

x_d = IrisHRT.array(x)
y_d = IrisHRT.array(y)
IrisHRT.parallel_for(N, axpy, alpha, x_d, y_d; flush=[x_d])

IrisHRT.synchronize()

axpy_test(alpha, x_ref, y)

@testset "axpy_test" begin
    @test x_ref == x
end

IrisHRT.finalize()
