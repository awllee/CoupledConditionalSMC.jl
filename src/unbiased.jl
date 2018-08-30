using ProgressMeter

function unbiasedEstimate(model::SMCModel, lM::F1, h::F2,
  ccsmcio::CCSMCIO{Particle}, b::Int64, algorithm::Symbol = :BS,
  independentInitialization::Bool = false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64)) where {F1<:Function, F2<:Function, Particle}

  initializeCCSMC(model, lM, ccsmcio, algorithm, independentInitialization)

  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  v = h(ref1) # just to get the type of v

  for i in 1:maxit
    ccXpf!(model, lM, ccsmcio, algorithm, rngCouple)
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
  b::Int64, independentInitialization::Bool = false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64)) where {F<:Function, Particle}
  return unbiasedEstimate(model, error, h, ccsmcio, b, :AT,
    independentInitialization, rngCouple, maxit)
end

function unbiasedEstimates(model::SMCModel, lM::F1, h::F2,
  ccsmcio::CCSMCIO{Particle}, b::Int64, m::Int64, algorithm::Symbol = :BS,
  independentInitialization::Bool = false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64)) where {F1<:Function, F2<:Function, Particle}

  # just to get the type of v
  initializeCCSMC(model, ccsmcio, independentInitialization)
  v = h(ccsmcio.ref1)
  T = typeof(v)

  iterations = Vector{Int64}(undef, m)
  values = Vector{T}(undef, m)

  @showprogress 10 for i in 1:m
    v = unbiasedEstimate(model, lM, h, ccsmcio, b, algorithm,
      independentInitialization, rngCouple, maxit)
    iterations[i] = v[1]
    values[i] = v[2]
  end

  return iterations, values
end

function unbiasedEstimates(model::SMCModel, h::F, ccsmcio::CCSMCIO{Particle},
  b::Int64, m::Int64, independentInitialization::Bool = false,
  rngCouple::Bool = false, maxit::Int64 = typemax(Int64)) where
  {F<:Function, Particle}
  return unbiasedEstimates(model, error, h, ccsmcio, b, m, :AT,
    independentInitialization, rngCouple, maxit)
end
