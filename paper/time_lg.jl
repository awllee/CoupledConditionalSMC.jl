import RData
using RNGPool
using CoupledConditionalSMC
using BenchmarkTools

# data generated with A = 0.95 (not 0.9)
dataJLS = RData.load("demo/ar1data.RData")
ys = vec(dataJLS["observations"])

include("../demo/lgModel.jl")

# they start at time 0 with observations from time 1,2,...
# so we have an initial distribution of N(0,1+0.9^2)
theta = LGTheta(0.9, 1.0, 1.0, 1.0, 0.0, 1.81)
model = makeLGModel(theta, ys)
lM = LinearGaussian.makelM(theta)

function foo(model, lM, ccsmcio, algorithm)
  for i in 1:ccsmcio.n
    ccsmcio.ref1[i].x = -1.0
    ccsmcio.ref2[i].x = 1.0
  end
  ccXpf!(model, lM, ccsmcio, algorithm, true)
end

n = 500
N = 512
ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, n)

algorithm = :AT
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :AS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :BS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

n = 1000
N = 1024
ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, n)

algorithm = :AT
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :AS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :BS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)
