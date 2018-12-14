using RNGPool
using RData
using CoupledConditionalSMC
import Statistics: mean, var

dataJLS = RData.load("demo/ar1data.RData")
ys = vec(dataJLS["observations"])

include("../demo/lgModel.jl")

# they start at time 0 with observations from time 1,2,...
# so we have an initial distribution of N(0,1+0.9^2)
theta = LGTheta(0.9, 1.0, 1.0, 1.0, 0.0, 1.81)
model = makeLGModel(theta, ys)
lM = LinearGaussian.makelM(theta)
ko = kalman(theta, ys)


setRNGs(12345)

function runDemoBS(n::Int64, N::Int64, m::Int64, maxcost::Int64)
    maxiter = Int(round(maxcost/N))
    results = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m,
                                                 :BS, true, true, maxiter)
    results[results .== 0] .= maxiter
    results.*N
end

function runDemoAS(n::Int64, N::Float64, m::Int64, maxcost::Int64)
    N_ = Int(ceil(N*n))
    maxiter = Int(round(maxcost/N_))
    results = CoupledConditionalSMC.couplingTimes(model, lM, N_, n, m,
                                                  :AS, true, true, maxiter)
    results[results .== 0] .= maxiter
    results.*N_
end

function runDemoAT(n::Int64, N::Float64, m::Int64, maxcost::Int64)
  # Scale number of particles proportional to length
  N_ = Int(ceil(N*n))
  maxiter = Int(round(maxcost/N_))
  results = CoupledConditionalSMC.couplingTimes(model, N_, n, m,
                                                true, true, maxiter)
  results[results .== 0] .= maxiter
  results.*N_
end

# Number of replications
Nrepl = 10_000
# Time series lengths
ts = collect(100:50:300)
# Number of particles
Ns = 2 .^ (2:6)

BS_cost = zeros(length(ts), length(Ns), Nrepl)
AS_cost = deepcopy(BS_cost)

#Threads.@threads
for i in length(ts):-1:1
  t = ts[i]
  for j in 1:length(Ns)
    N = Ns[j]
    # BS estimator with given number of particles:
    BS_cost[i,j,:] = runDemoBS(t, N, Nrepl, 50000)
    # AT estimator with t*N/10 particles:
    #AT_cost[i,j,:] = runDemoAT(t, N/2.0, Nrepl, 50000)
    AS_cost[i,j,:] = runDemoAS(t, N/5.0, Nrepl, 50000)
  end
end

using JLD2
@save "paper/output/lgDemoCompareOptimalCost_results.jld2" BS_cost AS_cost ts Nrepl Ns
