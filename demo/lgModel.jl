using RNGPool
using SMCExamples.LinearGaussian
import SMCExamples.LinearGaussian: LGTheta, simulateLGModel, kalman,
  makeLGModel
import SMCExamples.Particles.Float64Particle

function setupLGModel(n::Int64, obsVariance::Float64)
  theta = LGTheta(1.0, 1.0, 1.0, obsVariance, 0.0, 1.0)
  ys = simulateLGModel(theta, n)
  model = makeLGModel(theta, ys)
  lM = LinearGaussian.makelM(theta)
  ko = kalman(theta, ys)
  return model, lM, ko
end
