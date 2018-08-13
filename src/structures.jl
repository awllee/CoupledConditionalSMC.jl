import SequentialMonteCarlo.SMCModel
import SimpleSMC.SMCIO

struct CCSMCIO{Particle, ParticleScratch}
  N::Int64
  n::Int64
  smcio1::SMCIO{Particle, ParticleScratch}
  smcio2::SMCIO{Particle, ParticleScratch}
  ref1::Vector{Particle}
  ref2::Vector{Particle}
  # normalized weights
  nws1::Vector{Float64}
  nws2::Vector{Float64}
  # common weights & probability
  cws::Vector{Float64}
  # residual weights
  rws1::Vector{Float64}
  rws2::Vector{Float64}
  # backward sampling weights
  bws1::Vector{Float64}
  bws2::Vector{Float64}
end

function CCSMCIO{Particle, ParticleScratch}(N::Int64, n::Int64) where
  {Particle, ParticleScratch}
  smcio1 = SMCIO{Particle, ParticleScratch}(N, n)
  smcio2 = SMCIO{Particle, ParticleScratch}(N, n)
  ref1 = Vector{Particle}(undef, n)
  ref2 = Vector{Particle}(undef, n)
  for i in 1:n
    ref1[i] = Particle()
    ref2[i] = Particle()
  end
  nws1 = Vector{Float64}(undef, N)
  nws2 = Vector{Float64}(undef, N)
  cws = Vector{Float64}(undef, N)
  rws1 = Vector{Float64}(undef, N)
  rws2 = Vector{Float64}(undef, N)
  bws1 = Vector{Float64}(undef, N)
  bws2 = Vector{Float64}(undef, N)
  return CCSMCIO{Particle, ParticleScratch}(N, n, smcio1, smcio2, ref1, ref2,
    nws1, nws2, cws, rws1, rws2, bws1, bws2)
end
