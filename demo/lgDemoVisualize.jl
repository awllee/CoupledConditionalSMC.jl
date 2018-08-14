using RNGPool

include("demo.jl")
include("lgModel.jl")

setRNGs(12345)

println("Using a linear Gaussian Model with n = 100, obsVariance = 1.0")
model, lM, _ = setupLGModel(100, 1.0)
visualizeCCSMC(model, lM, 128, 10000)
visualizeCCSMC(model, 128, 10000)

println("\nUsing a linear Gaussian Model with n = 1000, obsVariance = 100.0")
model, lM, _ = setupLGModel(1000, 100.0)
visualizeCCSMC(model, lM, 512, 100)
visualizeCCSMC(model, 512, 100)
