using SimpleSMC
using ProgressMeter

function _coupledResampleJLS!(ccsmcio::CCSMCIO)
  _computeWeights(ccsmcio, ccsmcio.smcio1.ws, ccsmcio.smcio2.ws)
  rng = getRNG()
  cProb = sum(ccsmcio.cws)

  N::Int64 = ccsmcio.N
  nCommon::Int64 = sampleBinomial(N-1, cProb, rng)

  as1 = ccsmcio.smcio1.internal.as
  as2 = ccsmcio.smcio2.internal.as
  scratch1 = ccsmcio.smcio1.internal.scratch1
  scratch2 = ccsmcio.smcio1.internal.scratch2

  @inbounds as1[1] = 1
  _sampleCategoricalSorted!(as1, ccsmcio.cws, nCommon, scratch1, scratch2, 1,
    rng)

  for i in 1:1+nCommon
    @inbounds as2[i] = as1[i]
  end

  if nCommon < N - 1
    ## rng-coupled multinomial on residuals appears to have no significant effect
    extraRNG = ccsmcio.extraRNG
    copyto!(extraRNG, rng)
    _sampleCategoricalSorted!(as1, ccsmcio.rws1, N-nCommon-1, scratch1,
      scratch2, 1+nCommon, rng)
    _sampleCategoricalSorted!(as2, ccsmcio.rws2, N-nCommon-1, scratch1,
      scratch2, 1+nCommon, extraRNG)
  end

  ## shuffling appears to have no significant effect
  σ = ccsmcio.σ
  idxs = ccsmcio.idxs
  randperm!(rng, σ)
  idxs .= as1
  for i in 2:N
    @inbounds as1[i] = idxs[σ[i-1]+1]
  end
  idxs .= as2
  for i in 2:N
    @inbounds as2[i] = idxs[σ[i-1]+1]
  end
end

function _ccXpfJLS!(model::SMCModel, ccsmcio::CCSMCIO{Particle}, lM::F) where
  {Particle, F<:Function}
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

  for p = 1:ccsmcio.n
    if p > 1
      _copyParticles!(zetaAncs1, zetas1, as1)
      _copyParticles!(zetaAncs2, zetas2, as2)
    else
    end

    _rngCoupledMutateParticles!(zetas1, zetas2, model.M!, p, zetaAncs1,
      zetaAncs2, pScratch, ccsmcio.ref1[p], ccsmcio.ref2[p], ccsmcio.extraRNG)

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
      _coupledResampleJLS!(ccsmcio)
      lM != error && _ancestorSample!(ccsmcio, p, lM)
    end

    _intermediateOutput!(ccsmcio.smcio1, p)
    _intermediateOutput!(ccsmcio.smcio2, p)
  end
end

function ccXpfJLS!(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  algorithm::Symbol = :BS) where {F<:Function, Particle}
  if algorithm == :AS
    _ccXpfJLS!(model, ccsmcio, lM)
    _pickParticles!(ccsmcio)
  else
    _ccXpfJLS!(model, ccsmcio, error)
    if algorithm == :BS
      _pickParticlesBS!(ccsmcio, lM)
    else
      _pickParticles!(ccsmcio)
    end
  end
end

function ccXpfJLS!(model::SMCModel, ccsmcio::CCSMCIO{Particle}) where Particle
  ccXpfJLS!(model, error, ccsmcio, :AT)
end

function couplingTimeJLS(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  algorithm::Symbol = :BS, independentInitialization::Bool = false,
  maxit::Int64 = typemax(Int64)) where {F<:Function, Particle}

  initializeCCSMC(model, lM, ccsmcio, algorithm, independentInitialization)

  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  for i in 1:maxit
    ccXpfJLS!(model, lM, ccsmcio, algorithm)
    checkEqual(ref1, ref2) && return i
  end
  return 0
end

function couplingTimeJLS(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  independentInitialization::Bool=false, maxit::Int64 = typemax(Int64)) where
  Particle
  return couplingTimeJLS(model, error, ccsmcio, :AT, independentInitialization,
    maxit)
end

function couplingTimesJLS(model::SMCModel, lM::F, N::Int64, n::Int64,
  m::Int64, algorithm::Symbol = :BS, independentInitialization::Bool=false,
  maxit::Int64 = typemax(Int64)) where {F<:Function, Particle}
  nt = Threads.nthreads()
  ccsmcios::Vector{CCSMCIO{model.particle, model.pScratch}} =
    Vector{CCSMCIO{model.particle, model.pScratch}}(undef, nt)
  Threads.@threads for i in 1:nt
    ccsmcios[i] = CCSMCIO{model.particle, model.pScratch}(N, n)
  end
  vs::Vector{Int64} = Vector{Int64}(undef, m)
  p = Progress(div(m, nt), 10)
  Threads.@threads for i in 1:m
    tid = Threads.threadid()
    vs[i] = couplingTimeJLS(model, lM, ccsmcios[tid], algorithm,
      independentInitialization, maxit)
    tid == 1 && update!(p, i)
  end
  return vs
end

function couplingTimesJLS(model::SMCModel, N::Int64, n::Int64, m::Int64,
  independentInitialization::Bool = false, maxit::Int64 = typemax(Int64)) where
  Particle
  return couplingTimesJLS(model, error, N, n, m, :AT,
    independentInitialization, maxit)
end
