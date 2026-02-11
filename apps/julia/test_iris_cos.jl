include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using CUDA
using AMDGPU
using Test

IrisHRT.init()

function iris_cos_cuda(N, y, x)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	@inbounds y[i] = CUDA.cos(x[i])
	return nothing
end

function iris_cos_hip(N, y, x)
	i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
	@inbounds y[i] = Base.cos(x[i])
	return nothing
end

function main()
	N = 8
	# Generate random vectors x and y of length N for the interval [0, 100]
	x = ones(Float32, N)
	y = ones(Float32, N)

	x_d = IrisHRT.array(x)
	y_d = IrisHRT.array(y)
	IrisHRT.task(
			in=[x], out=[y],
			gws=[N],
			wait=true, 
			kernel="iris_cos", 
			args=[N, y_d, x_d])
	println("Y: ", y_d)
    IrisHRT.synchronize()
end
main()

IrisHRT.finalize()

