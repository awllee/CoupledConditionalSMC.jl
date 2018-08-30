using RNGPool

include("demo.jl")
include("lgModel.jl")

import Statistics: mean, var

setRNGs(12345)

println("Using a LG Model with n = 100")
model, lM, _ = setupLGModel(100, 1.0)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(32, model.maxn)
vs = CoupledConditionalSMC.couplingTimes(model, lM, ccsmcio, 100, :BS, true)
println(mean(vs), ", ", var(vs))

println("Using a LG Model with n = 200")
model, lM, _ = setupLGModel(200, 1.0)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(32, model.maxn)
vs = CoupledConditionalSMC.couplingTimes(model, lM, ccsmcio, 100, :BS, true)
println(mean(vs), ", ", var(vs))

println("Using a LG Model with n = 400")
model, lM, _ = setupLGModel(400, 1.0)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(32, model.maxn)
vs = CoupledConditionalSMC.couplingTimes(model, lM, ccsmcio, 100, :BS, true)
println(mean(vs), ", ", var(vs))
