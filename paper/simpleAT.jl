using SequentialMonteCarlo
using RNGPool
using CoupledConditionalSMC
import SMCExamples.Particles.Float64Particle

@inline function simplelM(::Int64, particle::Float64Particle,
  newParticle::Float64Particle, ::Nothing)
  x::Float64 = particle.x
  y::Float64 = newParticle.x
  t::Float64 = x - y
  return -0.5 * t * t
end

function setupSimpleModel(n::Int64, boxSize::Float64)
  @inline function lG(::Int64, particle::Float64Particle, ::Nothing)
    return abs(particle.x) < boxSize ? 0.0 : -Inf
  end
  @inline function M!(newParticle::Float64Particle, rng::RNG, p::Int64,
    particle::Float64Particle, ::Nothing)
    if p == 1
      newParticle.x = randn(rng)
    else
      newParticle.x = particle.x + randn(rng)
    end
  end
  return SMCModel(M!, lG, n, Float64Particle, Nothing), simplelM
end

model, lM = setupSimpleModel(10000, 5.0)

import Statistics: mean, std

setRNGs(12345)

m = 1000

using DataFrames
box_at_df = DataFrame(n=Int[], N=Int[], mean=Union{Float64, Missing}[],
  std=Union{Float64, Missing}[])

ns = 200:200:1600
Ns = round.(Int64, 100*(ns/100).^(1.4))

for i in 1:length(ns)
  n = ns[i]
  N = Ns[i]
  println(n, " ", N)
  vs = CoupledConditionalSMC.couplingTimes(model, N, n, m, true, true)
  println(mean(vs))
  push!(box_at_df, [n, N, mean(vs), std(vs)])
end

using JLD2
@save "paper/box_at.jld2" box_at_df

println(box_at_df)

box_all_df = DataFrame(n=Int[], N=Int[], type=String[], mean=Union{Float64, Missing}[],
  std=Union{Float64, Missing}[])

ns = 200:200:1600
Ns = round.(Int64, 128*(ns/200))

for i in 1:length(ns)
  n = ns[i]
  N = Ns[i]
  println(n, " ", N)
  vs = CoupledConditionalSMC.couplingTimes(model, N, n, m, true, true)
  println(mean(vs))
  push!(box_all_df, [n, N, "AT", mean(vs), std(vs)])
  vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :AS, true, true)
  println(mean(vs))
  push!(box_all_df, [n, N, "AS", mean(vs), std(vs)])
  vs = CoupledConditionalSMC.couplingTimes(model, lM, 128, n, m, :BS, true, true)
  println(mean(vs))
  push!(box_all_df, [n, 128, "BS", mean(vs), std(vs)])
end

using JLD2
@save "paper/box_all.jld2" box_all_df

println(box_all_df)
