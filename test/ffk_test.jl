import SequentialMonteCarlo.SMCModel
using SMCExamples.FiniteFeynmanKac
import SMCExamples.Particles.Int64Particle

setRNGs(12345)

function _getFreqs(model::SMCModel, lM::F, ancestorSampling::Bool,
  ccsmcio::CCSMCIO, m::Int64, d::Int64) where F<:Function
  ccsmcio.ref1 .= FiniteFeynmanKac.Int642Path(1, d, ccsmcio.n)
  ccsmcio.ref2 .= FiniteFeynmanKac.Int642Path(d^ccsmcio.n, d, ccsmcio.n)
  counts1 = zeros(Int64, d^ccsmcio.n)
  counts2 = zeros(Int64, d^ccsmcio.n)
  result1 = Vector{Float64}(undef, d^ccsmcio.n)
  result2 = Vector{Float64}(undef, d^ccsmcio.n)
  for i = 1:m
    if lM != error
      ccXpf!(model, lM, ccsmcio, ancestorSampling)
    else
      ccXpf!(model, ccsmcio)
    end

    counts1[FiniteFeynmanKac.Path2Int64(ccsmcio.ref1, d)] += 1
    counts2[FiniteFeynmanKac.Path2Int64(ccsmcio.ref2, d)] += 1
  end
  result1 .= counts1 ./ m
  result2 .= counts2 ./ m
  return result1, result2
end

function testccsmc()
  d = 4
  n = 4
  ffk = FiniteFeynmanKac.randomFiniteFK(d, n)

  model = FiniteFeynmanKac.makeSMCModel(ffk)
  lM = FiniteFeynmanKac.makelM(ffk)
  densities = Vector{Float64}(undef, d^n)
  for i = 1:length(densities)
    densities[i] = FiniteFeynmanKac.fullDensity(ffk, FiniteFeynmanKac.Int642Path(i, d, n))
  end
  densities ./= sum(densities)

  nsamples = 2^12

  ccsmcio = CCSMCIO{model.particle, model.pScratch}(4, model.maxn)
  freqs1, freqs2 = _getFreqs(model, lM, false, ccsmcio, nsamples, d)

  testapproxequal(freqs1, densities, 0.05, false)
  testapproxequal(freqs2, densities, 0.05, false)

  freqs1, freqs2 = _getFreqs(model, error, false, ccsmcio, nsamples, d)

  testapproxequal(freqs1, densities, 0.05, false)
  testapproxequal(freqs2, densities, 0.05, false)

  freqs1, freqs2 = _getFreqs(model, lM, true, ccsmcio, nsamples, d)

  testapproxequal(freqs1, densities, 0.05, false)
  testapproxequal(freqs2, densities, 0.05, false)
end

@time @testset "Finite FK: ccsmc" begin
  testccsmc()
end
