using SimpleSMC
using ProgressMeter

function _ccXpf!(model::SMCModel, ccsmcio::CCSMCIO{Particle}, lM::F,
  rngCouple::Bool) where {Particle, F<:Function}
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
    end

    if rngCouple
      _rngCoupledMutateParticles!(zetas1, zetas2, model.M!, p, zetaAncs1,
        zetaAncs2, pScratch, ccsmcio.ref1[p], ccsmcio.ref2[p], ccsmcio.extraRNG)
    else
      _indexCoupledMutateParticles!(zetas1, zetas2, model.M!, p, zetaAncs1,
        zetaAncs2, pScratch, ccsmcio.ref1[p], ccsmcio.ref2[p])
    end

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
  algorithm::Symbol = :BS, rngCouple::Bool = false) where {F<:Function, Particle}
  if algorithm == :AS
    _ccXpf!(model, ccsmcio, lM, rngCouple)
    _pickParticles!(ccsmcio)
  else
    _ccXpf!(model, ccsmcio, error, rngCouple)
    if algorithm == :BS
      _pickParticlesBS!(ccsmcio, lM)
    else
      _pickParticles!(ccsmcio)
    end
  end
end

function ccXpf!(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  rngCouple::Bool = false) where Particle
  ccXpf!(model, error, ccsmcio, :AT, rngCouple)
end

function initializeCCSMC(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  algorithm::Symbol = :BS, independent::Bool = false) where
  {F<:Function, Particle}
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
  if algorithm == :BS || algorithm == :AS
    SimpleSMC.csmc!(model, lM, smcio1, ref1, ref1)
  else
    SimpleSMC.csmc!(model, smcio1, ref1, ref1)
  end
end

function initializeCCSMC(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  independent::Bool = false) where Particle
  initializeCCSMC(model, error, ccsmcio, :AT, independent)
end
