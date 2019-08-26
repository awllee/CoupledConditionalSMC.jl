using JLD2
using StatsPlots
using DataFrames

@load "paper/box_all.jld2" box_all_df
@load "paper/box_at.jld2" box_at_df

@df box_at_df scatter(:n, :mean)
p = @df box_all_df plot(:n, :mean, group=:type, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks=200:200:1600, size=(450,250))

savefig(p, "paper/simpleComparisonATASBS.pdf")
