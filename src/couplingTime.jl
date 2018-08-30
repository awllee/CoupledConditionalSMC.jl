using ProgressMeter

function couplingTime(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  algorithm::Symbol = :BS, independentInitialization::Bool = false,
  rngCouple::Bool = false, maxit::Int64 = typemax(Int64)) where
  {F<:Function, Particle}

  initializeCCSMC(model, lM, ccsmcio, algorithm, independentInitialization)

  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  for i in 1:maxit
    ccXpf!(model, lM, ccsmcio, algorithm, rngCouple)
    checkEqual(ref1, ref2) && return i
  end
  return 0
end

function couplingTime(model::SMCModel, ccsmcio::CCSMCIO{Particle},
  independentInitialization::Bool=false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64)) where Particle
  return couplingTime(model, error, ccsmcio, :AT, independentInitialization,
    rngCouple, maxit)
end

function couplingTimes(model::SMCModel, lM::F, ccsmcio::CCSMCIO{Particle},
  m::Int64, algorithm::Symbol = :BS, independentInitialization::Bool=false,
  rngCouple::Bool = false, maxit::Int64 = typemax(Int64)) where
  {F<:Function, Particle}
  vs::Vector{Int64} = Vector{Int64}(undef, m)
  @showprogress 10 for i in 1:m
    vs[i] = couplingTime(model, lM, ccsmcio, algorithm,
      independentInitialization, rngCouple, maxit)
  end
  return vs
end

function couplingTimes(model::SMCModel, ccsmcio::CCSMCIO{Particle}, m::Int64,
  independentInitialization::Bool = false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64)) where Particle
  return couplingTimes(model, error, ccsmcio, m, :AT, independentInitialization,
    rngCouple, maxit)
end
