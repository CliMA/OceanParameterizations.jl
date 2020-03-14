module ClimateSurrogates

using DiffEqFlux

export
    weights, bias,
    GP, construct_gpr, prediction


include("utils.jl")
include("gaussian_process.jl")

end # module
