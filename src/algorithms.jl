using SimpleSMC
using ProgressMeter

function _ccXpf!(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  lM::F = error) where {Particle, F<:Function}
  zetas1 = ccsmcio.smcio1.zetas
  zetaAncs1 = ccsmcio.smcio1.internal.zetaAncs
  lws1 = ccsmcio.smcio1.internal.lws
  ws1 = ccsmcio.smcio1.ws
  as1 = ccsmcio.smcio1.internal.as
  pScratch = ccsmcio.smcio1.internal.particleScratch
  logZhats1 = ccsmcio.smcio1.logZhats
  lZ1 = 0.0

  zetas2 = ccsmcio.smcio2.zetas
  zetaAncs2 = ccsmcio.smcio2.internal.zetaAncs
  lws2 = ccsmcio.smcio2.internal.lws
  ws2 = ccsmcio.smcio2.ws
  as2 = ccsmcio.smcio2.internal.as
  logZhats2 = ccsmcio.smcio2.logZhats
  lZ2 = 0.0

  rng = getRNG()

  for p = 1:ccsmcio.n
    if p > 1
      _copyParticles!(zetaAncs1, zetas1, as1)
      _copyParticles!(zetaAncs2, zetas2, as2)
    end

    _indexCoupledMutateParticles!(zetas1, zetas2, model.M!, p, zetaAncs1,
      zetaAncs2, pScratch, ccsmcio.ref1[p], ccsmcio.ref2[p], rng)

    _logWeightParticles!(lws1, p, model.lG, zetas1, pScratch)
    _logWeightParticles!(lws2, p, model.lG, zetas2, pScratch)

    maxlw::Float64 = maximum(lws1)
    ws1 .= exp.(lws1 .- maxlw)
    mws::Float64 = mean(ws1)
    lZ1 += maxlw + log(mws)
    ws1 ./= mws
    @inbounds logZhats1[p] = lZ1

    maxlw = maximum(lws2)
    ws2 .= exp.(lws2 .- maxlw)
    mws = mean(ws2)
    lZ2 += maxlw + log(mws)
    ws2 ./= mws
    @inbounds logZhats2[p] = lZ2

    if p < ccsmcio.n
      _coupledResample!(ccsmcio)
      lM != error && _ancestorSample!(ccsmcio, p, lM)
    end

    _intermediateOutput!(ccsmcio.smcio1, p)
    _intermediateOutput!(ccsmcio.smcio2, p)
  end
end

function ccXpf!(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  ancestorSampling::Bool = false) where
  {F<:Function, Particle}
  if ancestorSampling
    _ccXpf!(model, ccsmcio, lM)
    _pickParticles!(ccsmcio)
  else
    _ccXpf!(model, ccsmcio)
    if lM != error
      _pickParticlesBS!(ccsmcio, lM)
    else
      _pickParticles!(ccsmcio)
    end
  end
end

function ccXpf!(model::SMCModel, ccsmcio::CCSMCIO{Particle}) where Particle
  ccXpf!(model, error, ccsmcio, false)
end

function initializeCCSMC(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  independent::Bool = false) where {F<:Function, Particle}
  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  smcio1 = ccsmcio.smcio1
  smcio2 = ccsmcio.smcio2

  SimpleSMC.smc!(model, smcio1)
  SimpleSMC.pickParticle!(ref1, smcio1)

  if independent
    SimpleSMC.smc!(model, smcio2)
    SimpleSMC.pickParticle!(ref2, smcio2)
  else
    SimpleSMC._copyParticles!(ref2, ref1)
  end
  if lM != error
    SimpleSMC.csmc!(model, lM, smcio1, ref1, ref1)
  else
    SimpleSMC.csmc!(model, smcio1, ref1, ref1)
  end
end

function initializeCCSMC(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  independent::Bool = false) where Particle
  initializeCCSMC(model, error, ccsmcio, independent)
end

function checkEqual(v1::Vector{Particle}, v2::Vector{Particle}) where Particle
  for i = 1:length(v1)
    if v1[i] != v2[i]
      return false
    end
  end
  return true
end

function unbiasedEstimate(model::SMCModel, lM::F1, h::F2,
  ccsmcio::CCSMCIO{Particle}, b::Int64, independentInitialization::Bool,
  maxit::Int64 = typemax(Int64), ancestorSampling::Bool = false) where
  {F1<:Function, F2<:Function, Particle}

  initializeCCSMC(model, lM, ccsmcio, independentInitialization)

  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  v = h(ref1) # just to get the type of v

  for i in 1:maxit
    ccXpf!(model, lM, ccsmcio, ancestorSampling)
    if i >= b
      if i == b
        v = h(ref1)
      else
        v += h(ref1) - h(ref2)
      end
      if checkEqual(ref1, ref2)
        return (i, v)
      end
    end
  end
  # error("maxit iterations exceeded")
  return (maxit, v)
end

function unbiasedEstimate(model::SMCModel, h::F, ccsmcio::CCSMCIO{Particle},
  b::Int64, independentInitialization::Bool, maxit::Int64 = typemax(Int64)) where
  {F<:Function, Particle}
  return unbiasedEstimate(model, error, h, ccsmcio, b,
    independentInitialization, maxit)
end

function unbiasedEstimates(model::SMCModel, lM::F1, h::F2,
  ccsmcio::CCSMCIO{Particle}, b::Int64, m::Int64,
  independentInitialization::Bool, maxit::Int64 = typemax(Int64),
  ancestorSampling::Bool = false) where {F1<:Function, F2<:Function, Particle}

  # just to get the type of v
  initializeCCSMC(model, ccsmcio, independentInitialization)
  v = h(ccsmcio.ref1)
  T = typeof(v)

  iterations = Vector{Int64}(undef, m)
  values = Vector{T}(undef, m)

  @showprogress 10 for i in 1:m
    v = unbiasedEstimate(model, lM, h, ccsmcio, b, independentInitialization,
      maxit, ancestorSampling)
    iterations[i] = v[1]
    values[i] = v[2]
  end

  return iterations, values
end

function unbiasedEstimates(model::SMCModel, h::F, ccsmcio::CCSMCIO{Particle},
  b::Int64, m::Int64, independentInitialization::Bool,
  maxit::Int64 = typemax(Int64)) where {F<:Function, Particle}
  return unbiasedEstimates(model, error, h, ccsmcio, b, m,
    independentInitialization, maxit)
end

function couplingTime(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  independentInitialization::Bool, maxit::Int64 = typemax(Int64),
  ancestorSampling::Bool = false) where {F<:Function, Particle}

  initializeCCSMC(model, lM, ccsmcio, independentInitialization)

  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  for i in 1:maxit
    ccXpf!(model, lM, ccsmcio, ancestorSampling)
    checkEqual(ref1, ref2) && return i
  end
  return 0
end

function couplingTime(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  independentInitialization::Bool, maxit::Int64 = typemax(Int64)) where Particle

  return couplingTime(model, error, ccsmcio, independentInitialization, maxit)
end
