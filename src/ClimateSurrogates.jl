module ClimateSurrogates

# using DiffEqFlux

export
    coarse_grain, Dᶠ, Dᶜ,
    GaussianProcess, predict, uncertainty, SquaredExponential

include("coarse_graining.jl")
include("differentiation_operators.jl")
include("gaussian_process.jl")

end # module
