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

model, lM = setupSimpleModel(10000, 10.0)

import Statistics: mean, std

setRNGs(12345)

m = 1000

using DataFrames
box_df = DataFrame(n=Int[], N=Int[], Type=String[], mean=Union{Float64, Missing}[],
  std=Union{Float64, Missing}[])

for n in [500, 1000, 2000, 4000]
  for k in 4:7
    N = 2^k
    println(n, " ", N)
    maxit = 10*n
    vs = CoupledConditionalSMC.couplingTimes(model, N, n, m, true, true, maxit)
    push!(box_df, [n, N, "AT", mean(vs), std(vs)])
    vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :AS, true, true, maxit)
    push!(box_df, [n, N, "AS", mean(vs), std(vs)])
    vs = CoupledConditionalSMC.couplingTimes(model, lM, N, n, m, :BS, true, true, maxit)
    push!(box_df, [n, N, "BS", mean(vs), std(vs)])
  end
end

using JLD2
@save "paper/box.jld2" box_df

println(box_df)
