module FreeConvection

export
    # DimensionalData.jl dimensions
    zC, zF,

    # Utils
    coarse_grain, add_surface_fluxes,

    # Animations
    animate_data, animate_learned_heat_flux,

    # Training data
    FreeConvectionTrainingDataInput, rescale, wrangle_input_training_data, wrangle_output_training_data,

    # Neural differential equations
    FreeConvectionNDE, FreeConvectionNDEParameters, train_neural_differential_equation!,

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

const ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary"

const LESBRARY_DATA_DEPS = (
    DataDep("lesbrary_free_convection_1",
            "proto-LESbrary.jl free convection statistics (Qb = 5×10⁻⁷ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb5.0e-07_f1.0e-04_Nh256_Nz128_pilot2", "statistics.nc")),
    DataDep("lesbrary_free_convection_2",
            "proto-LESbrary.jl free convection statistics (Qb = 2.5×10⁻⁷ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb2.5e-07_f1.0e-04_Nh256_Nz128_pilot3", "statistics.nc")),
)

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end # module
