import SMCExamples.Particles.Float64Particle
using SMCExamples.LinearGaussian
import SMCExamples.LinearGaussian: LGTheta, simulateLGModel, kalman,
  makeLGModel
using RNGPool
using CoupledConditionalSMC

import Statistics: mean, var

function testCouplingTime(n::Int64, N::Int64, maxit::Int64)
  theta = LGTheta(1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
  ys = simulateLGModel(theta, n)

  model = makeLGModel(theta, ys)
  lM = LinearGaussian.makelM(theta)

  ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, model.maxn)

  timeBS = CoupledConditionalSMC.couplingTime(model, lM, ccsmcio, :BS)
  @test timeBS > 0
  timeAS = CoupledConditionalSMC.couplingTime(model, lM, ccsmcio, :AS)
  @test timeAS > 0
  timeAT = CoupledConditionalSMC.couplingTime(model, ccsmcio)
  @test timeAT > 0

  timesBS = CoupledConditionalSMC.couplingTimes(model, lM, N, model.maxn, 100,
    :BS)
  @test all(timesBS .> 0)
end

setRNGs(12345)

@time @testset "Time test" begin
  testCouplingTime(10, 16, 100000)
  testCouplingTime(20, 32, 100000)
end
