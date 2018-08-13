module CoupledConditionalSMC

using RNGPool
import Statistics.mean
using Random

include("structures.jl")
include("subroutines.jl")
include("algorithms.jl")

export CCSMCIO, ccXpf!, initializeCCSMC

end # module
