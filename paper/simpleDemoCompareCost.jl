using RNGPool
using CoupledConditionalSMC
import Statistics: mean, var, quantile
using JLD2

include("simpleModel.jl")

if length(ARGS) < 3
    n_rep = 100
    ofname = "simpleDemoCompareCost.jld2"
    task_id = 1
elseif length(ARGS) == 3
    n_rep = parse(Int64, ARGS[1])
    ofname = ARGS[2]
    task_id = parse(Int64, ARGS[3])
end

model, lM = setupSimpleModel(50000, 5.0)
setRNGs(12345 + task_id*n_rep)

function runDemo(algorithm, n::Int64, N::Int64, m::Int64;
    maxcost=1_000_000_000, maxmem::Int64=2^23)
    #N_ = Int(floor(min(n*N, maxmem)/n))
    #if N_<N
    #    println("Warning: Using N=", N_, " instead of N=", N)
    #end
    N_ = N
    maxiter = Int(floor(maxcost/N_/n))
    if algorithm == :BS || algorithm == :AS
        results = CoupledConditionalSMC.couplingTimes(model, lM, N_, n, m,
                                                    algorithm, true, true, maxiter)
    else
        results = CoupledConditionalSMC.couplingTimes(model, N_, n, m,
                                                      true, true, maxiter)
    end
    if all(ismissing.(results))
        results_ = ones(m)*Inf
    else
        results_ = results
    end
    #results.*N_/n
    results_, maxiter
end

# Time series lengths
ts = collect(1000:1000:5000)
Ns = 2 .^ (8:13)

BS_cost = zeros(length(ts), length(Ns), n_rep)
AS_cost = deepcopy(BS_cost)
maxIter = zeros(length(ts), length(Ns))

t0 = time()
#Threads.@threads
for i in length(ts):-1:1
  t = ts[i]
  for j in 1:length(Ns)
    N = Ns[j]
    println("BS: T=", t, ", N=", N)
    # BS estimator with given number of particles:
    BS_cost[i,j,:], _ = runDemo(:BS, t, N, n_rep)
    GC.gc()
    # AT estimator with t*N/10 particles:
    println("AS: T=", t, ", N=", N)
    AS_cost[i,j,:], maxIter[i,j] = runDemo(:AS, t, N, n_rep)
    GC.gc()
  end
end

elapsedTime = time()-t0

@save ofname BS_cost AS_cost ts n_rep Ns elapsedTime maxIter
