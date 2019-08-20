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

m = 1000

using DataFrames
jlsInd_df = DataFrame(n=Int[], N=Int[], Type=String[], mean=Union{Float64, Missing}[],
  std=Union{Float64, Missing}[])

for n in [50, 100, 200, 400, 800, 1600, 3200]
  for k in 6:10
    N = 2^k
    println(n, " ", N)
    vs = CoupledConditionalSMC.couplingTimes(model, N, n, m, true, false, 2000)
    push!(jlsInd_df, [n, N, "AT", mean(vs), std(vs)])
    vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :AS, true, false, 2000)
    push!(jlsInd_df, [n, N, "AS", mean(vs), std(vs)])
    vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :BS, true, false, 2000)
    push!(jlsInd_df, [n, N, "BS", mean(vs), std(vs)])
  end
end

using JLD2
@save "paper/jlsInd.jld2" jlsInd_df

println(jlsInd_df)
