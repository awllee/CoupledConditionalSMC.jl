using RNGPool

include("simpleModel.jl")
include("demo.jl")

setRNGs(12345)

## box is [-1.0,1.0], n = 1000
simpleModel = makeSimpleModel(1.0, 1000)

## CCBPF couples after a number of steps, less with larger values of N
visualizeCCSMC(simpleModel, simplelM, 2, 10000)
visualizeCCSMC(simpleModel, simplelM, 4, 10000)
visualizeCCSMC(simpleModel, simplelM, 8, 10000)
visualizeCCSMC(simpleModel, simplelM, 16, 10000)
visualizeCCSMC(simpleModel, simplelM, 32, 10000)
visualizeCCSMC(simpleModel, simplelM, 64, 10000)
visualizeCCSMC(simpleModel, simplelM, 128, 10000)

visualizeCCSMC(simpleModel, 128, 10000)
visualizeCCSMC(simpleModel, 2^14, 10000)


#
# ## with 128 particles, n = 10000, CCBPF couples much quicker
# visualize(10000, 16, true, 0.5, 100)
#
# ## with 2 particles, n = 1000, CCPF doesn't couple after many steps
# visualize(1000, 2, false, 0.5, 1000)
#
# ## with 1024 particles, n = 1000, CCPF does couple after a few steps
# visualize(1000, 512, false, 0.5, 1000)
#
# ## switch = 0.0001
#
# ## with 2 particles, n = 1000, CCBPF couples after many steps [or occasionally
# ## very quickly]
# visualize(10000, 2, true, 0.001, 10000)
#
# ## with 4 particles, n = 1000, CCBPF couples after many steps
# visualize(10000, 4, true, 0.001, 10000)
# visualize(10000, 8, true, 0.001, 10000)
# visualize(10000, 16, true, 0.001, 10000)
# visualize(10000, 32, true, 0.001, 10000)
# visualize(10000, 64, true, 0.001, 10000)
# visualize(10000, 128, true, 0.001, 10000)
#
#
#
# visualize(1000, 4, true, 0.5, 0.5, 0.5, 10000)
# visualize(1000, 4, true, 0.5, 0.5, 0.5, 10000)
#
# visualize(2000, 4, true, 0.0001, 10000)
# visualize(1000, 4, false, 0.5, 100)
#
# p0 = 0.5
# p1 = 0.5
# switch = 0.5
# n = 1000
# N = 1024
# model = makeBinaryModel(switch, p0, p1, n)
# lM = makeBinarylM(switch)
# ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, model.maxn)
# initializeCCSMC(model, lM, ccsmcio, true)
