import SimpleSMC: particleCopy!, _copyParticles!, _mutateParticles!,
  _logWeightParticles!, _intermediateOutput!, pickParticle!,
  _sampleCategoricalSorted!
import NonUniformRandomVariateGeneration: sampleCategorical, sampleBinomial


function _indexCoupledMutateParticles!(zetas1::Vector{Particle},
  zetas2::Vector{Particle}, M!::F, p::Int64, zetaAncs1::Vector{Particle},
  zetaAncs2::Vector{Particle}, pScratch::ParticleScratch, xref1::Particle,
  xref2::Particle) where {Particle, F<:Function, ParticleScratch}
  @inbounds particleCopy!(zetas1[1], xref1)
  @inbounds particleCopy!(zetas2[1], xref2)
  rng = getRNG()
  for j in 2:length(zetas1)
    @inbounds M!(zetas1[j], rng, p, zetaAncs1[j], pScratch)
    if p == 1 || zetaAncs1[j] == zetaAncs2[j]
      @inbounds particleCopy!(zetas2[j], zetas1[j])
    else
      @inbounds M!(zetas2[j], rng, p, zetaAncs2[j], pScratch)
    end
  end
end

function _rngCoupledMutateParticles!(zetas1::Vector{Particle},
  zetas2::Vector{Particle}, M!::F, p::Int64, zetaAncs1::Vector{Particle},
  zetaAncs2::Vector{Particle}, pScratch::ParticleScratch, xref1::Particle,
  xref2::Particle, extraRNG::RNG) where {Particle, F<:Function, ParticleScratch}
  @inbounds particleCopy!(zetas1[1], xref1)
  @inbounds particleCopy!(zetas2[1], xref2)
  rng = getRNG()
  copyto!(extraRNG, rng)
  for j in 2:length(zetas1)
    @inbounds M!(zetas1[j], rng, p, zetaAncs1[j], pScratch)
    @inbounds M!(zetas2[j], extraRNG, p, zetaAncs2[j], pScratch)
  end
end

function _computeWeights(ccsmcio::CCSMCIO, ws1::Vector{Float64},
  ws2::Vector{Float64})
  nws1 = ccsmcio.nws1
  nws2 = ccsmcio.nws2
  cws = ccsmcio.cws
  rws1 = ccsmcio.rws1
  rws2 = ccsmcio.rws2

  nws1 .= ws1
  nws1 ./= sum(nws1)

  nws2 .= ws2
  nws2 ./= sum(nws2)

  for i in 1:ccsmcio.N
    @inbounds cws[i] = min(nws1[i], nws2[i])
  end

  rws1 .= nws1 .- cws
  rws2 .= nws2 .- cws
end

function _coupledResample!(ccsmcio::CCSMCIO)
  _computeWeights(ccsmcio, ccsmcio.smcio1.ws, ccsmcio.smcio2.ws)
  rng = getRNG()
  cProb = sum(ccsmcio.cws)

  N::Int64 = ccsmcio.N
  nCommon::Int64 = sampleBinomial(N-1, cProb, rng)

  as1 = ccsmcio.smcio1.internal.as
  as2 = ccsmcio.smcio2.internal.as
  scratch1 = ccsmcio.smcio1.internal.scratch1
  scratch2 = ccsmcio.smcio1.internal.scratch2

  as1[1] = 1
  _sampleCategoricalSorted!(as1, ccsmcio.cws, nCommon, scratch1, scratch2, 1,
    rng)

  for i in 1:1+nCommon
    as2[i] = as1[i]
  end

  if nCommon < N - 1
    _sampleCategoricalSorted!(as1, ccsmcio.rws1, N-nCommon-1, scratch1,
      scratch2, 1+nCommon, rng)
    _sampleCategoricalSorted!(as2, ccsmcio.rws2, N-nCommon-1, scratch1,
      scratch2, 1+nCommon, rng)
  end

end

function _coupledSampleIndices(ccsmcio, ws1, ws2)
  _computeWeights(ccsmcio, ws1, ws2)
  rng = getRNG()
  cProb = sum(ccsmcio.cws)
  if rand(rng) < cProb
    idx = sampleCategorical(ccsmcio.cws, rng)
    return (idx, idx)
  else
    idx1 = sampleCategorical(ccsmcio.rws1, rng)
    idx2 = sampleCategorical(ccsmcio.rws2, rng)
    return (idx1, idx2)
  end
end

function _trace!(path::Vector{Particle}, smcio::SMCIO{Particle}, k) where Particle
  n::Int64 = smcio.n
  allAs::Vector{Vector{Int64}} = smcio.allAs
  allZetas::Vector{Vector{Particle}} = smcio.allZetas
  @inbounds particleCopy!(path[n], allZetas[n][k])
  for p = n-1:-1:1
    @inbounds k = allAs[p][k]
    @inbounds particleCopy!(path[p], allZetas[p][k])
  end
end

function _pickParticles!(ccsmcio)
  ws1 = ccsmcio.smcio1.ws
  ws2 = ccsmcio.smcio2.ws
  k1, k2 = _coupledSampleIndices(ccsmcio, ws1, ws2)
  _trace!(ccsmcio.ref1, ccsmcio.smcio1, k1)
  _trace!(ccsmcio.ref2, ccsmcio.smcio2, k2)
end

function _pickParticlesBS!(ccsmcio::CCSMCIO{Particle}, lM::F) where
  {Particle, F<:Function}
  allZetas1::Vector{Vector{Particle}} = ccsmcio.smcio1.allZetas
  allZetas2::Vector{Vector{Particle}} = ccsmcio.smcio2.allZetas
  allWs1::Vector{Vector{Float64}} = ccsmcio.smcio1.allWs
  allWs2::Vector{Vector{Float64}} = ccsmcio.smcio2.allWs
  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2
  n = ccsmcio.n
  N = ccsmcio.N
  pScratch = ccsmcio.smcio1.internal.particleScratch

  ws1 = ccsmcio.smcio1.ws
  ws2 = ccsmcio.smcio2.ws
  k1, k2 = _coupledSampleIndices(ccsmcio, ws1, ws2)
  particleCopy!(ref1[n], allZetas1[n][k1])
  particleCopy!(ref2[n], allZetas2[n][k2])

  bws1 = ccsmcio.bws1
  bws2 = ccsmcio.bws2

  for p=n-1:-1:1
    bws1 .= log.(allWs1[p])
    bws2 .= log.(allWs2[p])
    for j in 1:N
      bws1[j] += lM(p+1, allZetas1[p][j], ref1[p+1], pScratch)
      bws2[j] += lM(p+1, allZetas2[p][j], ref2[p+1], pScratch)
    end
    m = maximum(bws1)
    bws1 .= exp.(bws1 .- m)
    m = maximum(bws2)
    bws2 .= exp.(bws2 .- m)
    k1, k2 = _coupledSampleIndices(ccsmcio, bws1, bws2)
    particleCopy!(ref1[p], allZetas1[p][k1])
    particleCopy!(ref2[p], allZetas2[p][k2])
  end
end

## index-coupled ancestor sampling
## this is slightly different to
## Jacob, P.E., Lindsten, F. and Schön, T.B., 2018. Smoothing with couplings of
## conditional particle filters. J. of the Amer. Statist. Assoc. 2018.
## who appear to use quantile coupling of the ancestor index in their code
function _ancestorSample!(ccsmcio, p, lM::F) where F<:Function
  zetas1 = ccsmcio.smcio1.zetas
  zetas2 = ccsmcio.smcio2.zetas
  zetaRef1 = ccsmcio.ref1[p+1]
  zetaRef2 = ccsmcio.ref2[p+1]
  bws1 = ccsmcio.bws1
  bws2 = ccsmcio.bws2
  ws1 = ccsmcio.smcio1.ws
  ws2 = ccsmcio.smcio2.ws
  pScratch = ccsmcio.smcio1.internal.particleScratch

  N = ccsmcio.N

  bws1 .= log.(ws1)
  bws2 .= log.(ws2)
  for j in 1:N
    bws1[j] += lM(p+1, zetas1[j], zetaRef1, pScratch)
    bws2[j] += lM(p+1, zetas2[j], zetaRef2, pScratch)
  end
  m = maximum(bws1)
  bws1 .= exp.(bws1 .- m)
  m = maximum(bws2)
  bws2 .= exp.(bws2 .- m)
  k1, k2 = _coupledSampleIndices(ccsmcio, bws1, bws2)

  ccsmcio.smcio1.internal.as[1] = k1
  ccsmcio.smcio2.internal.as[1] = k2
end

function checkEqual(v1::Vector{Particle}, v2::Vector{Particle}) where Particle
  for i = 1:length(v1)
    if v1[i] != v2[i]
      return false
    end
  end
  return true
end
