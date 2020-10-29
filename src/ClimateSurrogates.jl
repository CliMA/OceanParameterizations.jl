module ClimateSurrogates

export
    coarse_grain, Dᶠ, Dᶜ,
    UnitMeanZeroVarianceScaling, MinMaxScaling, scale, unscale,
    GaussianProcess, predict, uncertainty, SquaredExponential,

    animate_variable, training_data

using Printf
using Statistics
using LinearAlgebra

using NCDatasets
using Plots

using Oceananigans.Utils

using Oceananigans.Grids: Cell, Face

# Gotta set this environment variable when using the GR run-time on Travis CI.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
# ENV["GKSwstype"] = "100"

include("coarse_graining.jl")
include("feature_scaling.jl")
include("differentiation_operators.jl")
include("gaussian_process.jl")

include("ocean_convection.jl")

end # module
