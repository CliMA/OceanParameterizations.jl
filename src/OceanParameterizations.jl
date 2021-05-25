module OceanParameterizations

if VERSION < v"1.5"
    error("OceanParameterizations.jl requires Julia v1.5 or newer.")
end

export
    # Utils
    coarse_grain, Dᶠ, Dᶜ, coarse_grain_linear_interpolation,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, scale, unscale,
    predict, animate_gif,

    # GaussianProcesses
    gp_model, best_kernel, get_kernel, euclidean_distance, derivative_distance, antiderivative_distance, error_per_gamma,

    # NeuralNetworks
    nn_model,

    # PhysicalParameterizations
    closure_kpp_full_evolution,
    closure_tke_full_evolution

using LinearAlgebra
using Printf
using Statistics
using Logging

using Flux
using JLD2
using NCDatasets
using Plots
using Oceananigans.Utils

using Oceananigans.Grids: Center, Face

import Base.inv

include("differentiation_operators.jl")
include("predict.jl")

# submodules
include("DataWrangling/DataWrangling.jl")
include("GaussianProcesses/GaussianProcesses.jl")
include("NeuralNetworks/NeuralNetworks.jl")
include("PhysicalParameterizations/PhysicalParameterizations.jl")

using .DataWrangling
using .GaussianProcesses
using .NeuralNetworks
using .PhysicalParameterizations

end # module
