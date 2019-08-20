import RData
using RNGPool
using CoupledConditionalSMC

include("demo.jl")
include("lgModel.jl")

# data generated with A = 0.95 (not 0.9)
dataJLS = RData.load("demo/ar1data.RData")
ys = vec(dataJLS["observations"])

# they start at time 0 with observations from time 1,2,...
# so we have an initial distribution of N(0,1+0.9^2)
theta = LGTheta(0.9, 1.0, 1.0, 1.0, 0.0, 1.81)
lM = LinearGaussian.makelM(theta)

setRNGs(12345)

# one can couple in a few hundred iterations using 4096 particles with n=500
# but with n=10000 one would require millions or billions of particles
# and this would exhaust main memory
model = makeLGModel(theta, ys[1:8000])
visualizeCCSMC(model, 2^12, true, true)
CoupledConditionalSMC.couplingTimes(model, 2^12, 4000, 10, true, true, 2000)

model = makeLGModel(theta, ys[1:500])
visualizeCCSMC(model, lM, 128, :BS, true, true)
visualizeCCSMC(model, lM, 128, :AS, true, true)
CoupledConditionalSMC.couplingTimes(model, lM, 128, 100, 10, :BS, true, true, 2000)
tmp = CoupledConditionalSMC.couplingTimes(model, lM, 128, 100, 100, :AS, true, true, 2000)
maximum(tmp)



model = makeLGModel(theta, ys[1:100])
CoupledConditionalSMC.couplingTimes(model, lM, 128, 100, 10, :BS, true, true, 2000)

model = makeLGModel(theta, ys[1:500])
visualizeCCSMC(model, lM, 128, :BS, true, true)
visualizeCCSMC(model, lM, 128, :AS, true, true)

println("\nUsing a linear Gaussian Model with n = 1000, obsVariance = 100.0")
model, lM, _ = setupLGModel(1000, 100.0)
visualizeCCSMC(model, lM, 512, :BS, true, false)
visualizeCCSMC(model, 512, true, false, 100)
visualizeCCSMC(model, lM, 512, :AS, true, false, 100)

visualizeCCSMC(model, lM, 512, :BS, true, true)
visualizeCCSMC(model, 512, true, true, 100)
visualizeCCSMC(model, lM, 512, :AS, true, true, 100)
