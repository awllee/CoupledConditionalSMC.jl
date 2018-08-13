using RNGPool
using Plots
Plots.gr()

include("demo.jl")

verbose = false

setRNGs(12345)
model, lM, _ = setupLGModel(1000, 1.0)
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
model, lM, _ = setupLGModel(1000, 100.0)
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
model, lM, _ = setupLGModel(1000, 10000.0)
@time b37 = CCSMCBoundaries(model, lM, 128, 1000, verbose); plot(b37)
@time b38 = CCSMCBoundaries(model, lM, 256, 1000, verbose); plot!(b38)
@time b39 = CCSMCBoundaries(model, lM, 512, 1000, verbose); plot!(b39)
@time b3x = CCSMCBoundaries(model, lM, 1024, 1000, verbose); plot!(b3x)
plot(b37)
plot!(b38)
plot!(b39)
plot!(b3x)
