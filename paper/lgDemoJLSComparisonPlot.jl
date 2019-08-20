using JLD2
using DataFrames
using StatsPlots

@load "paper/jls.jld2" jls_df

p = @df jls_df[jls_df[:, :Type] .== "AT", :] plot(:n, :mean, group=:N, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks = [50; 400; 800; 1600; 3200],
  size=(400,250))
Plots.savefig(p, "paper/lgDemoPlot_AT.pdf")
p = @df jls_df[jls_df[:, :Type] .== "AS", :] plot(:n, :mean, group=:N, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks = [50; 400; 800; 1600; 3200],
  size=(400,250))
Plots.savefig(p, "paper/lgDemoPlot_AS.pdf")
p = @df jls_df[jls_df[:, :Type] .== "BS", :] plot(:n, :mean, group=:N, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks = [50; 400; 800; 1600; 3200],
  size=(400,250))
Plots.savefig(p, "paper/lgDemoPlot_BS.pdf")

# @df jls_df[jls_df[:, :Type] .== "AT", :] scatter(:n, :std, group=:N)
# @df jls_df[jls_df[:, :Type] .== "AS", :] scatter(:n, :std, group=:N)
# @df jls_df[jls_df[:, :Type] .== "BS", :] scatter(:n, :std, group=:N)

@load "paper/jlsInd.jld2" jlsInd_df

p = @df jlsInd_df[jlsInd_df[:, :Type] .== "AT", :] plot(:n, :mean, group=:N, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks = [50; 400; 800; 1600; 3200],
  size=(400,250))
Plots.savefig(p, "paper/lgDemoIndPlot_AT.pdf")
p = @df jlsInd_df[jlsInd_df[:, :Type] .== "AS", :] plot(:n, :mean, group=:N, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks = [50; 400; 800; 1600; 3200],
  size=(400,250))
Plots.savefig(p, "paper/lgDemoIndPlot_AS.pdf")
p = @df jlsInd_df[jlsInd_df[:, :Type] .== "BS", :] plot(:n, :mean, group=:N, marker=4,
  xlabel="T", ylabel="mean coupling time", xticks = [50; 400; 800; 1600; 3200],
  size=(400,250))
Plots.savefig(p, "paper/lgDemoIndPlot_BS.pdf")
