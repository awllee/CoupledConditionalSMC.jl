import Statistics: quantile, mean
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

using JLD2
@load "paper/output/lgDemoCompareOptimalCost_results.jld2"

figure(figsize=(15,5))
beginprint("paper/output/lgDemoCompareOptimalCost.pdf")
plot_all_estimates(ts, [BS_cost AS_cost]; max_y=12e3)
endprint()
