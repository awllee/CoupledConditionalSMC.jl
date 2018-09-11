using RNGPool

include("demo.jl")
include("lgModel.jl")

setRNGs(12345)

println("Using a linear Gaussian Model with n = 100, obsVariance = 1.0")
model, lM, _ = setupLGModel(100, 1.0)
visualizeCCSMC(model, lM, 128, :BS, true, false)
visualizeCCSMC(model, 128, true, false)
visualizeCCSMC(model, lM, 128, :AS, true, false)

visualizeCCSMC(model, lM, 128, :BS, true, true)
visualizeCCSMC(model, 128, true, true)
visualizeCCSMC(model, lM, 128, :AS, true, true)

println("\nUsing a linear Gaussian Model with n = 1000, obsVariance = 100.0")
model, lM, _ = setupLGModel(1000, 100.0)
visualizeCCSMC(model, lM, 512, :BS, true, false)
visualizeCCSMC(model, 512, true, false, 100)
visualizeCCSMC(model, lM, 512, :AS, true, false, 100)

visualizeCCSMC(model, lM, 512, :BS, true, true)
visualizeCCSMC(model, 512, true, true, 100)
visualizeCCSMC(model, lM, 512, :AS, true, true, 100)
