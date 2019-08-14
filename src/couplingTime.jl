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

function couplingTimes(model::SMCModel, lM::F, N::Int64, n::Int64,
  m::Int64, algorithm::Symbol = :BS, independentInitialization::Bool=false,
  rngCouple::Bool = false, maxit::Int64 = typemax(Int64)) where
  {F<:Function, Particle}
  nt = Threads.nthreads()
  ccsmcios::Vector{CCSMCIO{model.particle, model.pScratch}} =
    Vector{CCSMCIO{model.particle, model.pScratch}}(undef, nt)
  Threads.@threads for i in 1:nt
    ccsmcios[i] = CCSMCIO{model.particle, model.pScratch}(N, n)
  end
  vs::Vector{Int64} = Vector{Int64}(undef, m)
  p = Progress(div(m, nt), 10)
  abort = false
  Threads.@threads for i in 1:m
    tid = Threads.threadid()
    abort && break
    vs[i] = couplingTime(model, lM, ccsmcios[tid], algorithm,
      independentInitialization, rngCouple, maxit)
    if vs[i] == 0
      abort = true
    end
    tid == 1 && update!(p, i)
  end
  if abort
    return [missing]
  else
    return vs
  end
end

function couplingTimes(model::SMCModel, N::Int64, n::Int64, m::Int64,
  independentInitialization::Bool = false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64)) where Particle
  return couplingTimes(model, error, N, n, m, :AT, independentInitialization,
    rngCouple, maxit)
end
