module ClimateSurrogatesTalk abou

export
    coarse_grain, Dᶠ, Dᶜ,
    UnitMeanZeroVarianceScaling, MinMaxScaling, scale, unscale,
    GaussianProcess, predict, uncertainty, SquaredExponential

include("coarse_graining.jl")
include("feature_scaling.jl")
include("differentiation_operators.jl")
include("gaussian_process.jl")

end # module
