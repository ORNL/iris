using CUDA

include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()

# 1 A simplified kernel: no start_col, just len_col
function saxpy2d_left_kernel_jacc(i, j, a, X, Y, len_col)
    @inbounds begin
        M, N = size(X)
        #@cuprintf(" MN: %d %d\n", M, N)
        # only need to check j < len_col now
        if i <= len_col && j <= N
            @inbounds Y[i, j] = a * X[i, j] + Y[i, j]
            #@cuprintf("  → i[%d] j[%d] Y = %d\n", i, j, M, N, X[i, j])
        end
    end
    return nothing
end

# 2 Usage example
function example_left()
    M, N = 128, 256
    L = 100
    M, N = 4, 4
    L = 2
    hX =  reshape(collect(1:M*N), M, N)
    hY =  reshape(collect(1:M*N), M, N)
    hZ = copy(hY)
    a  = 2

    len_col = L # means columns [0..99]
    
    dhX = IrisHRT.dmem(hX; region_outer_dim=true)
    dhZ = IrisHRT.dmem(hZ; region_outer_dim=true)
    
    IrisHRT.update_size_no_strides(dhX, [L, N])
    IrisHRT.update_size_no_strides(dhZ, [L, N])
    
    IrisHRT.parallel_for((len_col, N), Main.saxpy2d_left_kernel_jacc, a, dhX, dhZ, len_col; flush=[dhZ]) 

    println(Core.stdout, "X: ", hX)
    println(Core.stdout, "Y: ", hY)
    println(Core.stdout, "Z: ", hZ)
    println(Core.stdout, " result:", a*hX[1:L,1:N] + hY[1:L,1:N])
    @testset "saxpy2d_partial" begin
        @test isapprox(hZ[1:L, 1:N], a*hX[1:L,1:N] + hY[1:L,1:N]; atol=1e-6)
        @test hZ[L+1:M,1:N] ≈ hY[L+1:M,1:N]  # untouched
    end
    println("slice-0 saxpy2d OK!")
end

example_left()

IrisHRT.finalize()
