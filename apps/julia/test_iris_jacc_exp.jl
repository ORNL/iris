include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using CUDA
using Test

IrisHRT.init()

function iris_jacc_exp(i, y, x)
	@inbounds y[i] = Base.exp(x[i])
    return nothing
end


function main()
	N = Int64(8)
	# Generate random vectors x and y of length N for the interval [0, 100]
	x = ones(Float32, N)
	y = ones(Float32, N)

	x_d = IrisHRT.array(x)
	y_d = IrisHRT.array(y)
	IrisHRT.parallel_for(N, Main.iris_jacc_exp, y_d, x_d)
	println("Y: ", y_d)
    IrisHRT.synchronize()
    Y_host = IrisHRT.valid_host(y_d)
    @test Y_host == Float32[2.7182817, 2.7182817, 2.7182817, 2.7182817, 2.7182817, 2.7182817, 2.7182817, 2.7182817]
end
@testset "iris_jacc_exp" begin
    main()
end
IrisHRT.finalize()

