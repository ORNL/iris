using CUDA

include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

# 1 A simplified kernel: no start_col, just len_col
function saxpy2d_left_kernel_jacc(i, j, a, X, Y, len_col)
    @inbounds begin
        nrows = size(X,1)
        # only need to check j < len_col now
        if 0 ≤ i < nrows && j < len_col
            @inbounds Y[i+1, j+1] = a * X[i+1, j+1] + Y[i+1, j+1]
        end
    end
    return nothing
end

# 2 Usage example
function example_left()
    M, N = 128, 256
    hX = rand(Float32, M, N)
    hY = rand(Float32, M, N)
    hZ = copy(hY)
    a  = 2.5f0

    len_col = 100  # means columns [0..99]
    IrisHRT.parallel_for((len_col, N), Main.saxpy2d_left_kernel_jacc, a, hX, hZ, len_col; flush=[hZ]) 

    @testset "saxpy2d_partial" begin
    @test isapprox(hZ[5, 50], a*hX[5,50] + hY[5,50]; atol=1e-6)
    @test hZ[5,150] ≈ hY[5,150]  # untouched
    end
    println("slice-0 saxpy2d OK!")
end

example_left()

IrisHRT.finalize()
