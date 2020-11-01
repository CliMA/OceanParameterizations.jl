module ClimateParameterizations

if VERSION < v"1.5"
    error("ClimateParameterizations.jl requires Julia v1.5 or newer.")
end

export
    # Utils
    coarse_grain, Dᶠ, Dᶜ,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, scale, unscale,
    GaussianProcess, predict, uncertainty, SquaredExponential,

    # Ocean convection
    nc_constant,
    FreeConvectionTrainingDataInput, rescale,
    FreeConvectionNDE, FreeConvectionNDEParameters, initial_condition,
    solve_free_convection_nde, free_convection_solution,
    animate_variable, convection_training_data, animate_learned_heat_flux

using LinearAlgebra
using Printf
using Statistics
using Logging

using DifferentialEquations
using Flux
using NCDatasets
using Plots
using Oceananigans.Utils

using Oceananigans: OceananigansLogger
using Oceananigans.Grids: Cell, Face
using DiffEqSensitivity: InterpolatingAdjoint, ZygoteVJP

import Base.inv

include("coarse_graining.jl")
include("feature_scaling.jl")
include("differentiation_operators.jl")
include("gaussian_process.jl")

include("ocean_convection.jl")

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end # module
