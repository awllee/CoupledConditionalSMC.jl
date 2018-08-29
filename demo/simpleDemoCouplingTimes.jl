using RNGPool

include("simpleModel.jl")
include("demo.jl")

import Statistics: mean, var

setRNGs(12345)

println("Using a Simple Model with n = 100")
simpleModel, simplelM = setupSimpleModel(100, 1.0)
ccsmcio = CCSMCIO{simpleModel.particle, simpleModel.pScratch}(32, simpleModel.maxn)
vs = CoupledConditionalSMC.couplingTimes(simpleModel, simplelM, ccsmcio, true,
  100, 100000, false)
println(mean(vs), ", ", var(vs))

println("Using a Simple Model with n = 200")
simpleModel, simplelM = setupSimpleModel(200, 1.0)
ccsmcio = CCSMCIO{simpleModel.particle, simpleModel.pScratch}(32, simpleModel.maxn)
vs = CoupledConditionalSMC.couplingTimes(simpleModel, simplelM, ccsmcio, true,
  100, 100000, false)
println(mean(vs), ", ", var(vs))

println("Using a Simple Model with n = 400")
simpleModel, simplelM = setupSimpleModel(400, 1.0)
ccsmcio = CCSMCIO{simpleModel.particle, simpleModel.pScratch}(32, simpleModel.maxn)
vs = CoupledConditionalSMC.couplingTimes(simpleModel, simplelM, ccsmcio, true,
  100, 100000, false)
println(mean(vs), ", ", var(vs))
