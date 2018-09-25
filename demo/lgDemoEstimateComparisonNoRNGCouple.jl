import RData
using RNGPool
using CoupledConditionalSMC

# data generated with A = 0.95 (not 0.9)
dataJLS = RData.load("ar1data.RData")
ys = vec(dataJLS["observations"])

include("lgModel.jl")

# they start at time 0 with observations from time 1,2,...
# so we have an initial distribution of N(0,1+0.9^2)
theta = LGTheta(0.9, 1.0, 1.0, 1.0, 0.0, 1.81)
model = makeLGModel(theta, ys)
lM = LinearGaussian.makelM(theta)

import Statistics: mean, std

setRNGs(12345)

const rngCouple = false

function runDemo(model, N, n, m, AT, AS, BS)
  println("\n-----\nLG Model with N = ", N, ", n = ", n)

  function h(path::Vector{Float64Particle})
    return path[1].x
  end

  ko = kalman(theta, ys[1:n])
  println("True value = ", ko.smoothingMeans[1])
  println()

  if AT
    println("Ancestral Tracing:")
    times, values = CoupledConditionalSMC.unbiasedEstimates(model, h, N, n, 1,
      m, true, rngCouple)
    println("times: ", mean(times), ", ", std(times))
    println("estimates: ", mean(values), ", ", std(values))
    println()
  end

  if AS
    println("Ancestor Sampling:")
    times, values = CoupledConditionalSMC.unbiasedEstimates(model, lM, h, N, n, 1,
      m, :AS, true, rngCouple)
    println("times: ", mean(times), ", ", std(times))
    println("estimates: ", mean(values), ", ", std(values))
    println()
  end

  if BS
    println("Backward Sampling:")
    times, values  = CoupledConditionalSMC.unbiasedEstimates(model, lM, h, N, n, 1,
      m, :BS, true, rngCouple)
    println("times: ", mean(times), ", ", std(times))
    println("estimates: ", mean(values), ", ", std(values))
    println()
  end

end

m = 1000

println("\nValues are presented as mean, one standard deviation\n")

runDemo(model, 64, 50, m, true, true, true)
runDemo(model, 128, 50, m, true, true, true)

runDemo(model, 128, 100, m, true, true, true)
runDemo(model, 256, 100, m, true, true, true)

runDemo(model, 256, 200, m, true, true, true)
runDemo(model, 512, 200, m, true, true, true)

runDemo(model, 512, 400, m, false, true, true)
runDemo(model, 1024, 400, m, false, true, true)

runDemo(model, 64, 800, m, false, false, true)
runDemo(model, 128, 800, m, false, false, true)
runDemo(model, 256, 800, m, false, false, true)
runDemo(model, 512, 800, m, false, false, true)
runDemo(model, 1024, 800, m, false, false, true)
runDemo(model, 2048, 800, m, false, false, true)
