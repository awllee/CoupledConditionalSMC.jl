import SMCExamples.Particles.Float64Particle
using SMCExamples.LinearGaussian
import SMCExamples.LinearGaussian: LGTheta, simulateLGModel, kalman,
  makeLGModel
using RNGPool
using CoupledConditionalSMC

import Statistics: mean, var

function testEstimate(n::Int64, N::Int64, b::Int64, m::Int64, maxit::Int64)
  theta = LGTheta(1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
  ys = simulateLGModel(theta, n)
  ko = kalman(theta, ys)

  model = makeLGModel(theta, ys)
  lM = LinearGaussian.makelM(theta)

  ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, model.maxn)

  function h(path::Vector{Float64Particle})
    return path[1].x
  end

  # just check the code runs
  CoupledConditionalSMC.initializeCCSMC(model, ccsmcio, true)
  # just check the code runs
  CoupledConditionalSMC.initializeCCSMC(model, lM, ccsmcio, true)

  # # just check the code runs
  # CoupledConditionalSMC.unbiasedEstimate(model, h, ccsmcio, b, maxit)

  trueValue = ko.smoothingMeans[1]

  resultsBS = CoupledConditionalSMC.unbiasedEstimates(model, lM, h, ccsmcio, b, m,
    maxit)
  @test mean(resultsBS[2]) ≈ trueValue atol=0.1
  Zval = (mean(resultsBS[2]) - trueValue)/sqrt(var(resultsBS[2])/m)
  @test abs(Zval) < 3

  results = CoupledConditionalSMC.unbiasedEstimates(model, h, ccsmcio, b, m, maxit)
  Zval = (mean(results[2]) - trueValue)/sqrt(var(results[2])/m)
  @test abs(Zval) < 3
end

setRNGs(12345)

@time @testset "Estimate test" begin
  testEstimate(10, 16, 1, 1000, 100000)
  testEstimate(20, 32, 1, 1000, 100000)
end
