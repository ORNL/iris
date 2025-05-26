include(ENV["IRIS"]*"/src/IrisHRT.jl")
using .IrisHRT
using Test

function test_reinterpret_kernel!()
    A = Float32[0.1f0 0.2f0 0.3f0 0.4f0 ; 
            0.5f0 0.6f0 0.7f0 0.8f0 ; 
            0.9f0 1.0f0 1.1f0 1.2f0 ; 
            1.3f0 1.4f0 1.5f0 1.6f0 ;
            1.7f0 1.8f0 1.9f0 2.0f0 ;
            3.0f0 3.1f0 3.2f0 3.3f0 ]

    A_d = IrisHRT.dmem(A)

    B = IrisHRT.reinterpret(ComplexF32, A_d)

    println(Core.stdout, "element type: ", eltype(A_d))
    println(Core.stdout, "element size: ", sizeof(eltype(A_d)))
    println(Core.stdout, "size: ", size(A_d))

    println(Core.stdout, "reinterpret element type: ", eltype(B))
    println(Core.stdout, "reinterpret element size: ", sizeof(eltype(B)))
    println(Core.stdout, "reinterpret size: ", size(B))

    @test eltype(B) == ComplexF32
    @test sizeof(eltype(B)) == 8
    @test size(B) == (3,4)
end



@testset "reinterpret" begin
    IrisHRT.init()

    test_reinterpret_kernel!()
    
    IrisHRT.finalize()
end
