module ClimateSurrogates

using DiffEqFlux

export
    weights, bias,
    GaussianProcess, predict


include("utils.jl")
include("gaussian_process.jl")

end # module
