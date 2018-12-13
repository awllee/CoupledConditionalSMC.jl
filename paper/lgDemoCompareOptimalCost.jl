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

import Statistics: quantile

using GR

function draw_empty(xmin, xmax, ymin, ymax; xmargin=0.0, ymargin=0.05, fontscale=5)
    xdiff = (xmax-xmin)*xmargin; ydiff = (ymax-ymin)*ymargin
    # Empty plot with right axes:
    setcharheight(fontscale*0.027)
    plot(; xlim=[xmin-xdiff,xmax+xdiff],
           ylim=[ymin-ydiff,ymax+ydiff], ylabel="Cost")
    nothing
end

function setcolors_grayscale(Lx)
    # Color 1 = black
    setcolorrep(1, 0.0, 0.0, 0.0)
    # From 2=white to darker, in cycles
    for k in 1:Lx
        gr = 1.0 - rem(2*(k-1)/Lx, 1.0)*0.8
        setcolorrep(k+1, gr, gr, gr)
    end
end

function boxbars(x, dx, Q_)
    #verrorbars(x, Q_[:,3], Q_[:,1], Q_[:,5])
    for k in 1:length(x)
        x_ = x[k]
        setlinecolorind(1); setmarkercolorind(1)
        setfillcolorind(k+1)
        polyline([x_,x_], [Q_[k,1], Q_[k,5]])
        fillrect(x_-dx,x_+dx,Q_[k,2],Q_[k,4])
        drawrect(x_-dx,x_+dx,Q_[k,2],Q_[k,4])
        polyline([x_-dx,x_+dx], Q_[k,3]*ones(2))
        polyline([x_-dx,x_+dx], Q_[k,1]*ones(2))
        polyline([x_-dx,x_+dx], Q_[k,5]*ones(2))
    end
end

function plot_all_estimates(xs, E; group_width=0.4, box_width=0.8,
    max_y=Inf)
    n_xs = length(xs)
    n_ns = size(E)[2]
    dx = (xs[2]-xs[1])*group_width/n_xs
    setfillintstyle(GR.INTSTYLE_SOLID)
    setmarkertype(GR.MARKERTYPE_SOLID_CIRCLE)
    setmarkersize(1.0)
    xmin = xs[1]-dx*n_ns/2; xmax = xs[end]+dx*(n_ns+2)/2
    ymin = Inf; ymax = -Inf
    Q_ = Vector(undef,n_xs)
    M_ = Vector(undef, n_xs)
    for k in 1:n_xs
        Q_[k] = mapslices(x -> quantile(x, [0.05,0.25,0.5,0.75,0.95]), E[k,:,:], dims=2)
        ymin = min(ymin, minimum(Q_[k]))
        ymax = max(ymax, maximum(Q_[k]))
    end
    ymax = min(max_y, ymax)
    draw_empty(xmin, xmax, ymin, ymax)
    setcolors_grayscale(n_ns)
    for k in 1:n_xs
        setlinecolorind(1); setmarkercolorind(1)
        #setfillcolorind(rem(n_xs-k,3)+2)
        xs_ = xs[k] .+ (collect(1:n_ns) .- n_ns/2)*dx
        boxbars(xs_, dx/2*box_width, Q_[k])
        # Add mean:
        M_ = vec(mapslices(x->mean(x), E[k,:,:], dims=2))
        setmarkertype(GR.MARKERTYPE_DIAGONAL_CROSS)
        polymarker(xs_, M_)
    end
end

figure(figsize=(15,5))
beginprint("paper/output/lgDemoCompareOptimalCost.pdf")
plot_all_estimates(ts, [BS_cost AS_cost]; max_y=12e3)
endprint()

using JLD2
@save "paper/output/lgDemoCompareOptimalCost_results.jld2" BS_cost AS_cost ts Nrepl Ns
