using Test
using OceanParameterizations
using Statistics

@testset "OceanParameterizations" begin
    include("test_feature_scaling.jl")
    include("test_coarse_graining.jl")
end