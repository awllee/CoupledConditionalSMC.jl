import RData
using RNGPool
using CoupledConditionalSMC

# data generated with A = 0.95 (not 0.9)
dataJLS = RData.load("demo/ar1data.RData")
ys = vec(dataJLS["observations"])

include("lgModel.jl")

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

# N = 128 T = 50 17.84 (17.13) 7.73 (5.11)
# N = 256 T = 100 13.16 (11.09) 7.59 (5.05)
# N = 512 T = 200 12.52 (10.64) 6.77 (3.85)
# N = 1024 T = 400 12.74 (10.96) 6.77 (3.47)
# N = 2048 T = 800 13.58 (9.56) 6.34 (2.95)

runDemo(model, 128, 50, 1000, true, true, true)
runDemo(model, 256, 100, 1000, true, true, true)
runDemo(model, 512, 200, 1000, true, true, true)
runDemo(model, 1024, 400, 1000, true, true, true)
runDemo(model, 2048, 800, 1000, true, true, true)
# runDemo(model, 256, 500, 100, false, true, true)
# runDemo(model, 256, 1000, 100, false, true, true)
# runDemo(model, 256, 2000, 100, false, true, true)

# runDemo(model, 200, 2000, 10, false, true, true)
# runDemo(model, 100, 2000, 1, false, true, true)
