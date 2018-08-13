using Plots
Plots.gr()

include("simpleModel.jl")
include("demo.jl")

verbose = false

setRNGs(12345)
model, lM = setupSimpleModel(1000, 1.0) # box is [-1.0,1.0], n = 1000
@time b11 = CCSMCBoundaries(model, lM, 2, 10000, verbose); plot(b11)
@time b12 = CCSMCBoundaries(model, lM, 4, 10000, verbose); plot!(b12)
@time b13 = CCSMCBoundaries(model, lM, 8, 10000, verbose); plot!(b13)
@time b14 = CCSMCBoundaries(model, lM, 16, 10000, verbose); plot!(b14)
@time b15 = CCSMCBoundaries(model, lM, 32, 10000, verbose); plot!(b15)
plot(b11)
plot!(b12)
plot!(b13)
plot!(b14)
plot!(b15)

setRNGs(12345)
model, lM = setupSimpleModel(1000, 5.0) # box is [-5.0,5.0], n = 1000
@time b21 = CCSMCBoundaries(model, lM, 2, 10000, verbose); plot(b21)
@time b22 = CCSMCBoundaries(model, lM, 4, 10000, verbose); plot!(b22)
@time b23 = CCSMCBoundaries(model, lM, 8, 10000, verbose); plot!(b23)
@time b24 = CCSMCBoundaries(model, lM, 16, 10000, verbose); plot!(b24)
@time b25 = CCSMCBoundaries(model, lM, 32, 10000, verbose); plot!(b25)
plot(b21)
plot!(b22)
plot!(b23)
plot!(b24)
plot!(b25)

setRNGs(12345)
model, lM = setupSimpleModel(10.0, 1000) # box is [-10.0,10.0], n = 1000
@time b35 = CCSMCBoundaries(model, lM, 32, 1000, verbose); plot(b35)
@time b36 = CCSMCBoundaries(model, lM, 64, 1000, verbose); plot!(b36)
@time b37 = CCSMCBoundaries(model, lM, 128, 1000, verbose); plot!(b37)
@time b38 = CCSMCBoundaries(model, lM, 256, 1000, verbose); plot!(b38)
plot(b35)
plot!(b36)
plot!(b37)
plot!(b38)
