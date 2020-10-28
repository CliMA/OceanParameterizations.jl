module ClimateSurrogates

# using DiffEqFlux

export
    coarse_grain,
    GaussianProcess, predict, uncertainty, SquaredExponential

include("utils.jl")
include("gaussian_process.jl")

end # module
