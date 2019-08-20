using JLD2
using DataFrames
using StatsPlots

@load "paper/box.jld2" box_df

p = @df box_df[.!ismissing.(box_df[:, :mean]), :] plot(:n, :mean, group=(:Type, :N), marker=4,
  xlabel="T", ylabel="mean coupling time", xticks=[500;1000;2000;4000], size=(400,250))
Plots.savefig(p, "paper/boxDemoPlot.pdf")

@load "paper/boxInd.jld2" boxInd_df

p = @df boxInd_df[.!ismissing.(boxInd_df[:, :mean]), :] plot(:n, :mean, group=(:Type, :N), marker=4,
  xlabel="T", ylabel="mean coupling time", xticks=[500;1000;2000;4000], size=(450,250))
Plots.savefig(p, "paper/boxDemoIndPlot.pdf")
