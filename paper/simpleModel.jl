using SequentialMonteCarlo
using RNGPool
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
