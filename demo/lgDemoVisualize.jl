using RNGPool

include("demo.jl")
include("lgModel.jl")

setRNGs(12345)

model, lM, _ = setupLGModel(100, 1.0)
visualizeCCSMC(model, lM, 128, 10000)
visualizeCCSMC(model, 128, 10000)

model, lM, _ = setupLGModel(1000, 100.0)
visualizeCCSMC(model, lM, 512, 100)
visualizeCCSMC(model, 512, 100)
