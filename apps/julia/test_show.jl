include(ENV["IRIS"]*"/src/IrisHRT.jl")
using .IrisHRT

IrisHRT.init()

s = [1 2 3 4; 5 6 7 8 ; 9 10 11 12 ; 13 14 15 16 ]

x = IrisHRT.dmem(@view s[3:4,3:4])

println(Core.stdout, "host: ", s)
println(Core.stdout, "dev: ", x)
println(Core.stdout, "dev[1,1]: ", x[1,1])

IrisHRT.finalize()
