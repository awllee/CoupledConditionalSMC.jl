using RNGPool

include("simpleModel.jl")
include("demo.jl")

import Statistics: mean, var

setRNGs(12345)

println("Using a Simple Model with n = 100")
model, lM = setupSimpleModel(100, 1.0)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(32, model.maxn)
vs = CoupledConditionalSMC.couplingTimes(model, lM, ccsmcio, 100, :BS, false)
println(mean(vs), ", ", var(vs))

println("Using a Simple Model with n = 200")
model, lM = setupSimpleModel(200, 1.0)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(32, model.maxn)
vs = CoupledConditionalSMC.couplingTimes(model, lM, ccsmcio, 100, :BS, false)
println(mean(vs), ", ", var(vs))

println("Using a Simple Model with n = 400")
model, lM = setupSimpleModel(400, 1.0)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(32, model.maxn)
vs = CoupledConditionalSMC.couplingTimes(model, lM, ccsmcio, 100, :BS, false)
println(mean(vs), ", ", var(vs))
