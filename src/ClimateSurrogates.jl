module ClimateSurrogates

export
    coarse_grain,
    GaussianProcess, predict, uncertainty,
    SquaredExponential

include("utils.jl")
include("gaussian_process.jl")
include("Layers.jl")
include("Diffusion.jl")

end # module
