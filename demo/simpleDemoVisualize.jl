using RNGPool

include("simpleModel.jl")
include("demo.jl")

setRNGs(12345)

println("Using a Simple Model with n = 1000, boxSize = 1.0")
simpleModel, simplelM = setupSimpleModel(1000, 1.0)

## CCBPF couples after a number of steps, less with larger values of N
visualizeCCSMC(simpleModel, simplelM, 2, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 2, :BS, true, true)
visualizeCCSMC(simpleModel, simplelM, 4, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 4, :BS, true, true)
visualizeCCSMC(simpleModel, simplelM, 8, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 8, :BS, true, true)
visualizeCCSMC(simpleModel, simplelM, 16, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 16, :BS, true, true)
visualizeCCSMC(simpleModel, simplelM, 32, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 32, :BS, true, true)
visualizeCCSMC(simpleModel, simplelM, 64, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 64, :BS, true, true)
visualizeCCSMC(simpleModel, simplelM, 128, :BS, true)
visualizeCCSMC(simpleModel, simplelM, 128, :BS, true, true)

## CCPF essentially will struggle to couple with 128 particles
visualizeCCSMC(simpleModel, 128, true, false, 100)
visualizeCCSMC(simpleModel, 128, true, true, 100)
## CCPF essentially will couple eventually in one shot with 2^14 particles
visualizeCCSMC(simpleModel, 2^14, true, false, 100)
visualizeCCSMC(simpleModel, 2^14, true, true, 100)

## CCAPF performs well for N ∈ {2,4,8} but not the rest
visualizeCCSMC(simpleModel, simplelM, 2, :AS, true)
visualizeCCSMC(simpleModel, simplelM, 2, :AS, true, true)
visualizeCCSMC(simpleModel, simplelM, 4, :AS, true)
visualizeCCSMC(simpleModel, simplelM, 4, :AS, true, true)
visualizeCCSMC(simpleModel, simplelM, 8, :AS, true)
visualizeCCSMC(simpleModel, simplelM, 8, :AS, true, true)
visualizeCCSMC(simpleModel, simplelM, 16, :AS, true, true, 1000)
visualizeCCSMC(simpleModel, simplelM, 32, :AS, true, true, 1000)
visualizeCCSMC(simpleModel, simplelM, 64, :AS, true, true, 1000)
visualizeCCSMC(simpleModel, simplelM, 128, :AS, true, true, 1000)
