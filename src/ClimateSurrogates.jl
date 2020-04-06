module ClimateSurrogates

using DiffEqFlux

export
    weights, bias,
    GaussianProcess, predict, uncertainty,
    SquaredExponential


include("utils.jl")
include("gaussian_process.jl")

end # module
