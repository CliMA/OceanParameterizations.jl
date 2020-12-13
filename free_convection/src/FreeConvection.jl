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

const ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary/free_convection_training_data/"

const LESBRARY_DATA_DEPS = (
    DataDep("free_convection_Qb1e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 1×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb1.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb1e-8", "statistics.nc"),
            "f935dbc46281c478141053673145b32551c1656921992fd81e25a467cea106ea"),
    DataDep("free_convection_Qb2e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 2×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb2.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb2e-8", "statistics.nc"),
            "9bad22e7ceb7f5bb8a562d222869b37ed331771c451af1a03fcafb23360e51ee"),
    DataDep("free_convection_Qb4e-8",
            "proto-LESbrary.jl free convection statistics (Qb = 4×10⁻⁸ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb4.0e-08_f1.0e-04_Nh256_Nz128_free_convection_Qb4e-8", "statistics.nc"),
            "2a7813826a5b1109983b7761971a584b0f78f49fd30fadb3a444c87e252a0bbd")
)

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end # module
