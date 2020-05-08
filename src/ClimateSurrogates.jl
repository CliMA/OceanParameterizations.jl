module ClimateSurrogates

# using DiffEqFlux

export
    weights, bias, coarse_grain,
    GaussianProcess, predict, uncertainty,
    SquaredExponential


include("utils.jl")
include("gaussian_process.jl")

end # module
