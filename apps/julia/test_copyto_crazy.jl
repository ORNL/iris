include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()
SIZE=10
submit_wait_flag=true
r = IrisHRT.fill(Float64, 0.5, SIZE)
r_old = IrisHRT.dmem(Float64, SIZE)
#println(Core.stdout, " r: ", r[1:min(SIZE,10)])
IrisHRT.copyto!(r_old, r; submit=submit_wait_flag, wait=submit_wait_flag)
r = r + r
IrisHRT.synchronize()

r_host = IrisHRT.valid_host(r, flush=true)
r_old_host = IrisHRT.valid_host(r_old, flush=true)

println(Core.stdout, " r:", r_host[1:min(SIZE,10)])
println(Core.stdout, " r_old: ", r_old_host[1:min(SIZE,10)])
r_ref_p5 = fill(Float64(0.5), SIZE)
r_ref_1 = fill(Float64(1.0), SIZE)

@testset "copyto_test" begin
    @test r_old_host == r_ref_p5
    @test r_host == r_ref_1
end

IrisHRT.finalize()
