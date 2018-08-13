using RNGPool
using CoupledConditionalSMC
import Statistics: mean, var

include("lgModel.jl")

function runDemo(n::Int64, N::Int64, b::Int64, m::Int64, maxit::Int64)
  model, lM, ko = setupLGModel(n, 1.0)

  ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, model.maxn)

  function h(path::Vector{Float64Particle})
    return path[1].x
  end

  println("\n-----\nTrue value = ", ko.smoothingMeans[1])

  println("\nBackward sampling")
  resultsBS = CoupledConditionalSMC.unbiasedEstimates(model, lM, h, ccsmcio, b,
    m, maxit)
  println("mean of estimates = ", mean(resultsBS[2]))
  println("mean no. iterations = ", mean(resultsBS[1]))
  println("estimated variance of estimator = ", var(resultsBS[2]))
  println("should be approx. standard normal: ",
    (mean(resultsBS[2]) - ko.smoothingMeans[1])/sqrt(var(resultsBS[2])/m))

  println("\nAncestral tracing")
  results = CoupledConditionalSMC.unbiasedEstimates(model, h, ccsmcio, b, m,
    maxit)
  println("mean of estimates = ", mean(results[2]))
  println("mean no. iterations = ", mean(results[1]))
  println("estimated variance of estimator = ", var(results[2]))
  println("should be approx. standard normal: ",
    (mean(results[2]) - ko.smoothingMeans[1])/sqrt(var(results[2])/m))
  println("-----\n")
end

setRNGs(12345)

runDemo(10, 16, 1, 1000, 100000)
runDemo(20, 16, 1, 1000, 100000)
runDemo(30, 32, 1, 1000, 100000)
