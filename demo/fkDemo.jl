using CoupledConditionalSMC
import SequentialMonteCarlo.SMCModel
using SMCExamples.FiniteFeynmanKac
import SMCExamples.Particles.Int64Particle
using RNGPool

# setRNGs(12345)

d = 2
n = 3
ffk = FiniteFeynmanKac.randomFiniteFK(d, n)

model = FiniteFeynmanKac.makeSMCModel(ffk)
densities = Vector{Float64}(undef, d^n)
for i = 1:length(densities)
  densities[i] = FiniteFeynmanKac.fullDensity(ffk, FiniteFeynmanKac.Int642Path(i, d, n))
end
densities ./= sum(densities)

lM = FiniteFeynmanKac.makelM(ffk)

function _getFreqs(model::SMCModel, lM::F, ccsmcio::CCSMCIO,
  algorithm::Symbol, m::Int64, d::Int64) where F<:Function
  ccsmcio.ref1 .= FiniteFeynmanKac.Int642Path(1, d, ccsmcio.n)
  ccsmcio.ref2 .= FiniteFeynmanKac.Int642Path(d^ccsmcio.n, d, ccsmcio.n)
  counts1 = zeros(Int64, d^ccsmcio.n)
  counts2 = zeros(Int64, d^ccsmcio.n)
  result1 = Vector{Float64}(undef, d^ccsmcio.n)
  result2 = Vector{Float64}(undef, d^ccsmcio.n)
  for i = 1:m
    ccXpf!(model, lM, ccsmcio, algorithm)
    counts1[FiniteFeynmanKac.Path2Int64(ccsmcio.ref1, d)] += 1
    counts2[FiniteFeynmanKac.Path2Int64(ccsmcio.ref2, d)] += 1
  end
  result1 .= counts1 ./ m
  result2 .= counts2 ./ m
  return result1, result2
end

nsamples = 2^22

# ccsmcio = CCSMCIO{model.particle, model.pScratch}(16, model.maxn)
ccsmcio = CCSMCIO{model.particle, model.pScratch}(4, model.maxn)
_getFreqs(model, error, ccsmcio, :AT, 4, d)
@time freqs1, freqs2 = _getFreqs(model, error, ccsmcio, :AT, nsamples, d)

println(sum(abs.(densities-freqs1)))
println(sum(abs.(densities-freqs2)))

_getFreqs(model, lM, ccsmcio, :BS, 4, d)
@time freqs1, freqs2 = _getFreqs(model, lM, ccsmcio, :BS, nsamples, d)

println(sum(abs.(densities-freqs1)))
println(sum(abs.(densities-freqs2)))

_getFreqs(model, lM, ccsmcio, :AS, 4, d)
@time freqs1, freqs2 = _getFreqs(model, lM, ccsmcio, :AS, nsamples, d)

println(sum(abs.(densities-freqs1)))
println(sum(abs.(densities-freqs2)))
