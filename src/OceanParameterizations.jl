module OceanParameterizations

if VERSION < v"1.5"
    error("OceanParameterizations.jl requires Julia v1.5 or newer.")
end

export
    # Utils
    coarse_grain, Dᶠ, Dᶜ,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, scale, unscale,
    predict, animate_gif,

    # GaussianProcesses
    gp_model, get_kernel, euclidean_distance, derivative_distance, antiderivative_distance,

    # NeuralNetworks
    nn_model,

    # PhysicalParameterizations
    closure_free_convection_kpp_full_evolution,
    closure_free_convection_tke_full_evolution

using LinearAlgebra
using Printf
using Statistics
using Logging

# using DifferentialEquations
using Flux
using JLD2
using NCDatasets
using Plots
using Oceananigans.Utils

using Oceananigans: OceananigansLogger
using Oceananigans.Grids: Cell, Face
using DiffEqSensitivity: InterpolatingAdjoint, ZygoteVJP

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
