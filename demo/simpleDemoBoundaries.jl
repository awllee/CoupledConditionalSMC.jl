using Plots
Plots.gr()

include("simpleModel.jl")
include("demo.jl")

setRNGs(12345)
model, lM = setupSimpleModel(1000, 1.0) # box is [-1.0,1.0], n = 1000
@time b11 = CCSMCBoundaries(model, lM, 2, :BS, 10000); plot(b11)
@time b12 = CCSMCBoundaries(model, lM, 4, :BS, 10000); plot!(b12)
@time b13 = CCSMCBoundaries(model, lM, 8, :BS, 10000); plot!(b13)
@time b14 = CCSMCBoundaries(model, lM, 16, :BS, 10000); plot!(b14)
@time b15 = CCSMCBoundaries(model, lM, 32, :BS, 10000); plot!(b15)
plot(b11, label="N=2")
plot!(b12, label="N=4")
plot!(b13, label="N=8")
plot!(b14, label="N=16")
plot!(b15, label="N=32")
xlabel!("iteration")
ylabel!("boundary")
title!("Boundary against iteration for boxSize = 1.0")
savefig("output/simpleDemoBoundaries1.pdf")

setRNGs(12345)
model, lM = setupSimpleModel(1000, 5.0) # box is [-5.0,5.0], n = 1000
@time b21 = CCSMCBoundaries(model, lM, 2, :BS, 10000); plot(b21)
@time b22 = CCSMCBoundaries(model, lM, 4, :BS, 10000); plot!(b22)
@time b23 = CCSMCBoundaries(model, lM, 8, :BS, 10000); plot!(b23)
@time b24 = CCSMCBoundaries(model, lM, 16, :BS, 10000); plot!(b24)
@time b25 = CCSMCBoundaries(model, lM, 32, :BS, 10000); plot!(b25)
plot(b21, label="N=2")
plot!(b22, label="N=4")
plot!(b23, label="N=8")
plot!(b24, label="N=16")
plot!(b25, label="N=32")
xlabel!("iteration")
ylabel!("boundary")
title!("Boundary against iteration for boxSize = 5.0")
savefig("output/simpleDemoBoundaries2.pdf")

setRNGs(12345)
model, lM = setupSimpleModel(1000, 10.0) # box is [-10.0,10.0], n = 1000
@time b35 = CCSMCBoundaries(model, lM, 32, :BS, 1000); plot(b35)
@time b36 = CCSMCBoundaries(model, lM, 64, :BS, 1000); plot!(b36)
@time b37 = CCSMCBoundaries(model, lM, 128, :BS, 1000); plot!(b37)
@time b38 = CCSMCBoundaries(model, lM, 256, :BS, 1000); plot!(b38)
plot(b35, label="N=32")
plot!(b36, label="N=64")
plot!(b37, label="N=128")
plot!(b38, label="N=256")
xlabel!("iteration")
ylabel!("boundary")
title!("Boundary against iteration for boxSize = 10.0")
savefig("output/simpleDemoBoundaries3.pdf")
