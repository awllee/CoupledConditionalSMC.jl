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

function checkBoundary(ref1::Vector{Float64Particle}, ref2::Vector{Float64Particle}) :: Int64
    b = 0
    for i in 1:length(ref1)
        (ref1[i].x != ref2[i].x) && break
        b = i
    end
    b
end

function couplingBoundary(model, lM, algorithm=:AS, N=2000, n=500, maxit=500,
    rngCouple = true, independentInitialisation = true)
    io = CCSMCIO{model.particle, model.pScratch}(N, n)
    initializeCCSMC(model, lM, io, algorithm, independentInitialisation)
    bd = fill(n, maxit)
    for i in 1:maxit
        ccXpf!(model, lM, io, algorithm, rngCouple)
        bd_ = checkBoundary(io.ref1, io.ref2)
        bd[i] = bd_
        (bd_ == n) && break
    end
    bd
end

setRNGs(12345)
N = 64    # Number of particles
n = 800     # Length of time series
maxit = 500 # Maximum number of iterations
bdAT = couplingBoundary(model, lM, :AT, N, n, maxit)
bdAS = couplingBoundary(model, lM, :AS, N, n, maxit)
bdBS = couplingBoundary(model, lM, :BS, N, n, maxit)

using GR
figure(figsize=(15,5))
beginprint("paper/output/lgDemoPaths.pdf")
GR.legend("BS", "AS", "ATT"; loc=2)
GR.xlabel("Iteration")
GR.ylabel("Coupling boundary")
GR.ylim([-30, n+30])
GR.xlim([0, maxit])
GR.plot(1:maxit, bdBS, "k-.", 1:maxit, bdAS, "k--", 1:maxit, bdAT, "k-")
endprint()
