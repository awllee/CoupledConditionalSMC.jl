module CoupledConditionalSMC

using RNGPool
import Statistics.mean
using Random

include("structures.jl")
include("subroutines.jl")
include("algorithms.jl")
include("unbiased.jl")
include("couplingTime.jl")
include("jls.jl")

export CCSMCIO, ccXpf!, initializeCCSMC

end # module
