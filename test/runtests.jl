using CoupledConditionalSMC
using RNGPool
using Test

function testapproxequal(a::Vector{Float64}, b::Vector{Float64}, tol::Float64,
  verbose::Bool)
  v::Float64 = maximum(abs.(a-b))
  verbose && (println(a) ; println(b))
  verbose && println("$v < $tol ?")
  @test v < tol
end

function testapproxequal(a::Float64, b::Float64, tol::Float64, verbose::Bool)
  v::Float64 = abs(a-b)
  verbose && println("$v < $tol ?")
  @test v < tol
end

include("ffk_test.jl")
include("estimate_test.jl")
include("couplingTime_test.jl")
