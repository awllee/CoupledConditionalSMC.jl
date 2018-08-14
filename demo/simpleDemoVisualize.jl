using RNGPool

include("simpleModel.jl")
include("demo.jl")

setRNGs(12345)

## box is [-1.0,1.0], n = 1000
simpleModel, simplelM = setupSimpleModel(1000, 1.0)

## CCBPF couples after a number of steps, less with larger values of N
visualizeCCSMC(simpleModel, simplelM, 2, 10000)
visualizeCCSMC(simpleModel, simplelM, 4, 10000)
visualizeCCSMC(simpleModel, simplelM, 8, 10000)
visualizeCCSMC(simpleModel, simplelM, 16, 10000)
visualizeCCSMC(simpleModel, simplelM, 32, 10000)
visualizeCCSMC(simpleModel, simplelM, 64, 10000)
visualizeCCSMC(simpleModel, simplelM, 128, 10000)

## CCPF essentially will struggle to couple with 128 particles
visualizeCCSMC(simpleModel, 128, 100)
## CCPF essentially will couple eventually in one shot with 2^14 particles
visualizeCCSMC(simpleModel, 2^14, 100)
