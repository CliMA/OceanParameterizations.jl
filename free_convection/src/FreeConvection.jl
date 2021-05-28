module FreeConvection

export
    # DimensionalData.jl dimensions
    zC, zF,

    # Utils
    coarse_grain, add_surface_fluxes!,

    # Animations
    animate_training_data, animate_learned_free_convection,

    # Training data
    FreeConvectionTrainingDataInput, rescale, wrangle_input_training_data, wrangle_output_training_data,

    # Neural differential equations
    FreeConvectionNDE, ConvectiveAdjustmentNDE, FreeConvectionNDEParameters, train_neural_differential_equation!, solve_nde,

    # Testing and comparisons
    oceananigans_convective_adjustment_nn, free_convection_kpp, free_convection_tke_mass_flux, optimize_kpp_parameters,

    # Testing
    compute_nde_solution_history, plot_epoch_loss, animate_nde_loss, plot_comparisons, plot_loss_matrix

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
using OceanParameterizations
using Oceananigans.Units

using DimensionalData: basetypeof
using GeoData: AbstractGeoStack, window
using Oceananigans: OceananigansLogger, Center, Face
using Oceananigans.Utils: prettytime

@dim zC ZDim "z"
@dim zF ZDim "z"

include("coarse_grain.jl")
include("add_surface_fluxes.jl")
include("animations.jl")
include("training_data.jl")
include("free_convection_nde.jl")
include("convective_adjustment_nde.jl")
include("solve.jl")
include("training.jl")
include("testing.jl")
include("k_profile_parameterization.jl")
include("tke_mass_flux.jl")
# include("optimize_kpp_parameters.jl")
include("oceananigans_nn.jl")

include("data_dependencies.jl")

function __init__()
    Logging.global_logger(OceananigansLogger())
end

# TODO: Add these definitions to Oceananigans!

using Oceananigans.OutputReaders: FieldTimeSeries, FieldDataset

Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

Base.getindex(fds::FieldDataset, inds...) = Base.getindex(fds.fields, inds...)

import Oceananigans: interior
using Oceananigans.Grids: topology, interior_parent_indices

interior(f::FieldTimeSeries{X, Y, Z}) where {X, Y, Z} =
    view(parent(f), interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
                    interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
                    interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz),
                    :)

import Oceananigans.OutputReaders: FieldTimeSeries, InMemory
using Oceananigans.Architectures: array_type
using Oceananigans.Grids: total_size, offset_data

function FieldTimeSeries(grid, location, times; architecture=CPU(), name="", filepath="", bcs=nothing)
    LX, LY, LZ = location

    Nt = length(times)
    data_size = total_size(location, grid)

    ArrayType = array_type(architecture)
    raw_data = zeros(data_size..., Nt) |> ArrayType
    data = offset_data(raw_data, grid, location)

    return FieldTimeSeries{LX, LY, LZ}(InMemory(), data, architecture, grid, bcs, times, name, filepath, 4)
end

export FieldTimeSeries

end # module
