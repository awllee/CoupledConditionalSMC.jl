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

import Statistics: mean, std

setRNGs(12345)

m = 1000

using DataFrames

lg_all_df = DataFrame(n=Int[], N=Int[], type=String[], mean=Union{Float64, Missing}[],
  std=Union{Float64, Missing}[])

ns = 200:200:1600
Ns = round.(Int64, 512*(ns/200))

for i in 1:length(ns)
  n = ns[i]
  N = Ns[i]
  println(n, " ", N)
  vs = CoupledConditionalSMC.couplingTimes(model, N, n, m, true, true)
  println(mean(vs))
  push!(lg_all_df, [n, N, "AT", mean(vs), std(vs)])
  vs = CoupledConditionalSMC.couplingTimes(model, lM, 512, n, m, :AS, true, true)
  println(mean(vs))
  push!(lg_all_df, [n, 512, "AS", mean(vs), std(vs)])
  vs = CoupledConditionalSMC.couplingTimes(model, lM, 512, n, m, :BS, true, true)
  println(mean(vs))
  push!(lg_all_df, [n, 512, "BS", mean(vs), std(vs)])
end

using JLD2
@save "paper/lg_all.jld2" lg_all_df

println(lg_all_df)

# using StatsPlots
# @df lg_all_df plot(:n, :mean, group=:type, marker=4)
