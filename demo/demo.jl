using CoupledConditionalSMC
import SequentialMonteCarlo.SMCModel

@inline function makeCharacter(v1::Vector{Particle}, v2::Vector{Particle},
  start::Int64, finish::Int64) where Particle
  for j in start:finish
    if v1[j] != v2[j]
      return "x"
    end
  end
  return "-"
end

function makeString(v1::Vector{Particle}, v2::Vector{Particle},
  quantum::Int64) where Particle
  s::String = ""
  n::Int64 = length(v1)
  for i = 1:n
    if mod(i, quantum) == 1 || quantum == 1
      start = i
      finish = min(i+quantum-1, n)
      s *= makeCharacter(v1, v2, start, finish)
    end
  end
  s *= "\n"
  return s
end

# run the CCxPF and visualize until the meeting time
function visualizeCCSMC(model::SMCModel, lM::F, N::Int64,
  algorithm::Symbol = :BS, independentInitialization::Bool = false,
  rngCouple::Bool = false, maxit::Int64 = typemax(Int64)) where F<:Function

  if algorithm == :BS
    println("\nBackward sampling, N = ", N, ":\n")
  elseif algorithm == :AS
    println("\nAncestor sampling, N = ", N, ":\n")
  else
    println("\nAncestral tracing, N = ", N, ":\n")
  end
  ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, model.maxn)

  initializeCCSMC(model, lM, ccsmcio, algorithm, independentInitialization)

  quantum = max(1, ceil(Int64, ccsmcio.n / displaysize(stdout)[2]))

  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2

  printstyled(makeString(ref1, ref2, quantum), color=:green)

  for i in 1:maxit
    if algorithm == :BS || algorithm == :AS
      ccXpf!(model, lM, ccsmcio, algorithm, rngCouple)
    else
      ccXpf!(model, ccsmcio, rngCouple)
    end
    printstyled(makeString(ref1, ref2, quantum), color=:green)
    if CoupledConditionalSMC.checkEqual(ref1, ref2)
      println("coupled at iteration $i")
      return i
    end
  end
  println("never coupled in $maxit iterations")
  return maxit
end

# run the CCPF and visualize until the meeting time
function visualizeCCSMC(model::SMCModel, N::Int64,
  independentInitialization::Bool = false, rngCouple::Bool = false,
  maxit::Int64 = typemax(Int64))
  return visualizeCCSMC(model, error, N, :AT, independentInitialization,
    rngCouple, maxit)
end

function computeBoundary(v1::Vector{Particle}, v2::Vector{Particle}) where Particle
  for i = 1:length(v1)
    if v1[i] != v2[i] return i-1 end
  end
  return length(v1)
end

# run the CCBPF and record the boundaries
function CCSMCBoundaries(model::SMCModel, lM::F, N::Int64,
  algorithm::Symbol, maxit::Int64) where F<:Function

  ccsmcio = CCSMCIO{model.particle, model.pScratch}(N, model.maxn)
  initializeCCSMC(model, lM, ccsmcio, algorithm, true)

  boundaries = Vector{Int64}(undef, maxit)
  ref1 = ccsmcio.ref1
  ref2 = ccsmcio.ref2
  n = length(ref1)

  maxb::Int64 = 1
  for i in 1:maxit
    ccXpf!(model, lM, ccsmcio, algorithm)
    boundary = computeBoundary(ref1, ref2)
    boundaries[i] = boundary
    if boundary == n
      return boundaries[1:i]
    end
  end
  return boundaries
end

function CCSMCBoundaries(model::SMCModel, N::Int64,
  maxit::Int64 = typemax(Int64))
  return CCSMCBoundaries(model, error, N, maxit, :AT)
end
