using Statistics
using Plots

rectangle(xmin, xmax, ymin, ymax) = Plots.Shape([xmin,xmax,xmax,xmin], [ymin,ymin,ymax,ymax])
function Plots_boxbars!(p, x, dx, Q_; boxColor=:lightgray)
    #verrorbars(x, Q_[:,3], Q_[:,1], Q_[:,5])
    for k in 1:length(x)
        x_ = x[k]
        Plots.plot!(p, [x_,x_], Q_[k,[1,5]], color="black")
        Plots.plot!(p, rectangle(x_-dx,x_+dx,Q_[k,2],Q_[k,4]), color=boxColor)
        Plots.plot!(p, [x_-dx,x_+dx], Q_[k,[3,3]], color="black")
        Plots.plot!(p, [x_-dx,x_+dx], Q_[k,[1,1]], color="black")
        Plots.plot!(p, [x_-dx,x_+dx], Q_[k,[5,5]], color="black")
    end
    nothing
end

function Plots_all_estimates!(p, xs, E; group_width=0.3, box_width=0.8,
    max_y=Inf, only_mean=false, markershape=:xcross, showmean=true,
    box_colors = (:lightgray, :lightyellow, :lightgreen, :pink, :lightblue, :cyan))
    n_colors = length(box_colors)
    function limits_tune(xmin, xmax; gap=0.05)
        local dx = xmax - xmin
        (xmin - dx*gap, xmax + dx*gap)
    end

    n_xs = length(xs)
    n_ns = size(E)[2]
    dx = (xs[2]-xs[1])*group_width/n_xs
    xmin = xs[1]-dx*n_ns/2; xmax = xs[end]+dx*(n_ns+2)/2
    ymin = Inf; ymax = -Inf
    # For now: just plug in max cost
    #E[E.==Inf] .= max_y
    Q_ = Vector(undef, n_ns)
    M_ = Vector(undef, n_ns)
    for k in 1:n_ns
        M_[k] = mapslices(x -> mean(x), E[:,k,:], dims=2)
        if ~only_mean
            Q_[k] = mapslices(x -> quantile(x, [0.05,0.25,0.5,0.75,0.95]), E[:,k,:], dims=2)
            ymin = min(ymin, minimum(Q_[k]))
            ymax = max(ymax, maximum(Q_[k]))
        end
    end
    ymax = min(max_y, ymax)

    Plots.plot!(p, xlim=(xmin-n_xs/2*dx, xmax), ylim=(0, ymax), #ylim=limits_tune(ymin, ymax),
    framestyle=:axes, gridlinewidht=3.0, gridstyle=:solid,
    gridalpha=0.2, xticks=xs)
    display(p)
    #result = input("foo")
    if ~only_mean
        for k in 1:n_xs
            xs_ = xs[k]
            Plots.vspan!(p, [xs_-dx*n_xs*1.3,xs_], color=:gray, fillalpha=0.3, linealpha=0)
        end
        for k in 1:n_ns
            col = box_colors[rem(k-1,n_colors)+1]
            xs_ = xs .+ (k .- .5 .- n_ns/2)*dx
            Plots_boxbars!(p, xs_, dx/2*box_width, Q_[k]; boxColor=col)
        end
    end
    if showmean
    for k in 1:n_ns
        xs_ = xs .+ (k .- .5 .- n_ns/2)*dx
        Plots.plot!(p, xs_, M_[k]; seriestype=:scatter, markershape = markershape,
        markercolor = :black, markersize=3)
    end
    end
    #for k in 1:n_xs
    #    col = box_colors[rem(k-1,n_colors)+1]
    #    xs_ = xs[k] .+ (collect(1:n_ns) .- n_ns/2)*dx
    #    Plots_boxbars!(p, xs_, dx/2*box_width, Q_[k]; boxColor=col)
        #M_ = vec(mapslices(x->mean(x), E[k,:,:], dims=2))
        #polymarker(xs_, M_)
    #end
    Plots.plot!(p, legend=false)
    nothing
end

function add_legend!(p, Ns, x_, dx, y, box_colors)
    n_Ns = length(Ns)
    dy = diff(y)[1]/n_Ns
    for i in 1:n_Ns
        y_ = y[2] - dy*(i-.5)
        Plots.plot!(p, rectangle(x_,x_+dx,y_-dy*.4, y_+dy*.4), color=box_colors[i])
        Plots.annotate!(p, x_+1.5*dx, y_, text(string("N=", Ns[i]), :left, 7))
    end
end

function plotCost(ts, AS_iter, BS_iter, Ns, maxIter, max_y=75)
    # Calculate 'normalised cost' = iter*N_particles/length_of_series
    n_ts = length(ts); n_Ns = length(Ns); n_rep = size(AS_iter)[3]
    BS_cost = zeros(n_ts, n_Ns, n_rep)
    AS_cost = zeros(n_ts, n_Ns, n_rep)
    BS_capped = zeros(n_ts, n_Ns)
    AS_capped = zeros(n_ts, n_Ns)
    BS_capval = zeros(n_ts, n_Ns)*NaN
    AS_capval = zeros(n_ts, n_Ns)*NaN
    for i in 1:n_ts
      t = ts[i]
      for j in 1:n_Ns
          N = Ns[j]
          maxCost_ = maxIter[i,j]*N/t
          BS_capped_ = isinf.(BS_iter[i,j,:])
          AS_capped_ = isinf.(AS_iter[i,j,:])
          BS_capped[i,j] = sum(BS_capped_)
          AS_capped[i,j] = sum(AS_capped_)
          BS_cost[i,j,:] .= BS_iter[i,j,:]*N/t
          AS_cost[i,j,:] .= AS_iter[i,j,:]*N/t
          BS_cost[i,j,BS_capped_] .= maxCost_
          AS_cost[i,j,AS_capped_] .= maxCost_
          if any(BS_capped_)
              BS_capval[i,j] = min(maxCost_, max_y)
          end
          if any(AS_capped_)
              AS_capval[i,j] = min(maxCost_, max_y)
          end
      end
    end
    #BS_cost = mapslices(x -> x./ts, mapslices(x -> x.*Ns, BS_iter, dims=2), dims=1)
    #AS_cost = mapslices(x -> x./ts, mapslices(x -> x.*Ns, AS_iter, dims=2), dims=1)
    #maxCost = mapslices(x -> x./ts, mapslices(x -> x.*Ns, maxIter, dims=2), dims=1)
    box_colors = (:lightgray, :cyan, :lightyellow, :lightgreen, :pink, :lightblue)

    p = Plots.plot()
    Plots_all_estimates!(p, ts, [BS_cost AS_cost], showmean=false, box_colors=box_colors)
    add_legend!(p, Ns, 4900,80, [45,75], box_colors)
    Plots_all_estimates!(p, ts, [BS_capval AS_capval]; only_mean=true, markershape=:star8)
    Plots.plot!(p, ylim=(0,max_y), xlabel="T", ylabel="Normalised cost")
    p, AS_capped, BS_capped
end

using JLD2
@load "simpleDemoCompareCostAll_results.jld2"
p, AS_capped, BS_capped = plotCost(ts, AS_iter, BS_iter, Ns, maxIter)
Plots.plot!(p, size=(700,200))
Plots.savefig("simpleDemoCompareCost.pdf")
