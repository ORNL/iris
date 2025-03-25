include(ENV["IRIS"] * "/include/iris/IrisHRT.jl")
using .IrisHRT
using CUDA

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
end
main()

IrisHRT.finalize()

