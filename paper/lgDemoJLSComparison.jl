import RData
using RNGPool
using CoupledConditionalSMC

# data generated with A = 0.95 (not 0.9)
dataJLS = RData.load("demo/ar1data.RData")
ys = vec(dataJLS["observations"])

include("../demo/lgModel.jl")

# they start at time 0 with observations from time 1,2,...
# so we have an initial distribution of N(0,1+0.9^2)
theta = LGTheta(0.9, 1.0, 1.0, 1.0, 0.0, 1.81)
model = makeLGModel(theta, ys)
lM = LinearGaussian.makelM(theta)
ko = kalman(theta, ys)

import Statistics: mean, std

setRNGs(12345)

function runDemo(model, N, n, m, AT, AS, BS)
  println("LG Model with N = ", N, ", n = ", n)

## JLS use independent initialization

  if AT
    println("Ancestral Tracing:")
    vs = CoupledConditionalSMC.couplingTimes(model, N, n, m, true, true)
    println(mean(vs), ", ", std(vs))
    println()

    println("JLS Ancestral Tracing:")
    vs = CoupledConditionalSMC.couplingTimesJLS(model, N, n, m, true)
    println(mean(vs), ", ", std(vs))
    println()
  end

  if AS
    println("Ancestor Sampling:")
    vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :AS, true, true)
    println(mean(vs), ", ", std(vs))
    println()

    println("JLS Ancestor Sampling:")
    vs = CoupledConditionalSMC.couplingTimesJLS(model, lM, N, n, m, :AS, true)
    println(mean(vs), ", ", std(vs))
    println()
  end

  if BS
    println("Backward Sampling:")
    vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :BS, true, true)
    println(mean(vs), ", ", std(vs))
    println()
  end

end

m = 1000

runDemo(model, 64, 50, m, true, true, true)
runDemo(model, 128, 50, m, true, true, true)

runDemo(model, 128, 100, m, true, true, true)
runDemo(model, 256, 100, m, true, true, true)

runDemo(model, 256, 200, m, true, true, true)
runDemo(model, 512, 200, m, true, true, true)

runDemo(model, 512, 400, m, true, true, true)
runDemo(model, 1024, 400, m, true, true, true)
