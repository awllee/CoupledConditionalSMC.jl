using RNGPool
using SequentialMonteCarlo
import SMCExamples.Particles.Float64Particle
using CoupledConditionalSMC
using BenchmarkTools

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


function foo(model, lM, ccsmcio, algorithm)
  for i in 1:ccsmcio.n
    ccsmcio.ref1[i].x = -1.0
    ccsmcio.ref2[i].x = 1.0
  end
  ccXpf!(model, lM, ccsmcio, algorithm, true)
end

n = 500
N = 512
ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, n)

algorithm = :AT
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :AS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :BS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

n = 1000
N = 1024
ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, n)

algorithm = :AT
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :AS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)

algorithm = :BS
initializeCCSMC(model, lM, ccsmcio, algorithm, true)
@btime foo(model, lM, ccsmcio, algorithm)
