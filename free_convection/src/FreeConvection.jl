module FreeConvection

export
    # DimensionalData.jl dimensions
    zC, zF,

    # Utils
    coarse_grain, add_surface_fluxes,

    # Animations
    animate_data, animate_learned_free_convection,

    # Training data
    FreeConvectionTrainingDataInput, rescale, wrangle_input_training_data, wrangle_output_training_data,

    # Neural differential equations
    FreeConvectionNDE, ConvectiveAdjustmentNDE, FreeConvectionNDEParameters, train_neural_differential_equation!,

    # Other parameterizations
    free_convection_kpp, optimize_kpp_parameters,

    # Testing
    compute_nde_solution_history, plot_epoch_loss, animate_nde_loss

using Logging
using Printf
using Statistics

using DataDeps
using DimensionalData
using GeoData
using NCDatasets
using Plots
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqFlux
using JLD2
using Oceananigans.Utils
using OceanParameterizations

using DimensionalData: basetypeof
using GeoData: AbstractGeoStack, window
using Oceananigans: OceananigansLogger, Cell, Face

@dim zC ZDim "z"
@dim zF ZDim "z"

include("coarse_grain.jl")
include("add_surface_fluxes.jl")
include("animations.jl")
include("training_data.jl")
include("free_convection_nde.jl")
include("convective_adjustment_nde.jl")
include("training.jl")
include("testing.jl")
include("kpp.jl")
include("optimize_kpp_parameters.jl")

include("data_dependencies.jl")

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end # module
