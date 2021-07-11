module FreeConvection

export
    # Utils
    coarse_grain, add_surface_fluxes!, add_convective_adjustment_flux!,

    # Animations
    animate_training_data, animate_learned_free_convection,

    # Training data
    FreeConvectionTrainingDataInput, rescale, wrangle_input_training_data, wrangle_output_training_data,

    # Neural differential equations
    FreeConvectionNDE, ConvectiveAdjustmentNDE, FreeConvectionNDEParameters, train_neural_differential_equation!, solve_nde,

    # Testing and comparisons
    oceananigans_convective_adjustment, oceananigans_convective_adjustment_with_neural_network,
    free_convection_kpp, free_convection_tke_mass_flux, optimize_kpp_parameters,

    # Testing
    compute_nde_solution_history,
    plot_epoch_loss, plot_epoch_loss_summary, plot_epoch_loss_summary_filled_curves,
    animate_nde_loss, plot_comparisons,
    plot_loss_matrix, plot_initial_vs_final_loss_matrix,

    # Data
    validate_simulation_ids, load_data

using Logging
using Printf
using Statistics

using DataDeps
using NCDatasets
using Plots
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqFlux
using JLD2

using OceanParameterizations
using Oceananigans.Units

using Oceananigans: OceananigansLogger, Center, Face
using Oceananigans.Utils: prettytime

using LinearAlgebra: Tridiagonal

using Oceananigans:
    CPU, RegularRectilinearGrid, CenterField, ZFaceField, ComputedField, set!, compute!, interior,
    FluxBoundaryCondition, GradientBoundaryCondition, TracerBoundaryConditions, fill_halo_regions!,
    Forcing, IncompressibleModel, Simulation, run!

using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.OutputWriters: NetCDFOutputWriter, TimeInterval
using Oceananigans.AbstractOperations: @at, ∂z

include("coarse_grain.jl")
include("add_surface_fluxes.jl")
include("convective_adjustment_flux.jl")
include("animations.jl")
include("training_data.jl")
include("free_convection_nde.jl")
include("convective_adjustment_nde.jl")
include("solve.jl")
include("training.jl")
include("testing.jl")
include("k_profile_parameterization.jl")
include("tke_mass_flux.jl")
include("oceananigans_nn.jl")
include("data.jl")

function __init__()
    Logging.global_logger(OceananigansLogger())

    @info "Registering data dependencies..."

    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    for dd in LESBRARY_DATA_DEPS
        DataDeps.register(dd)
    end
end

end # module
