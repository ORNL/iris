include(ENV["IRIS"] * "/src/IrisHRT.jl")
using .IrisHRT
using Test

IrisHRT.init()
SIZE=10
r = IrisHRT.fill(Float64, 0.5, SIZE)
r_old = IrisHRT.dmem(Float64, SIZE)

IrisHRT.copyto!(r_old, r)

r_new = IrisHRT.copy(r_old)

IrisHRT.synchronize()

r_new_host = IrisHRT.valid_host(r_new; flush=true)
r_old_host = IrisHRT.valid_host(r_old; flush=true)
r_host = IrisHRT.valid_host(r; flush=true)

println(Core.stdout, " r: ", r_host[1:min(SIZE,10)], " uid:", r.uid)
println(Core.stdout, " r_old: ", r_old_host[1:min(SIZE,10)], " uid:", r_old.uid)
println(Core.stdout, " r_new", r_new_host[1:min(SIZE,10)], " uid:", r_new.dmem.uid)

@testset "copyto" begin
    @test r_new_host == r_old_host
    @test r_old_host == r_host
end


IrisHRT.finalize()
