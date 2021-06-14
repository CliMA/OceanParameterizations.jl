module WindMixing

using JLD2: FILE_GROW_SIZE
export data, read_les_output,
       animate_prediction,
       mse, 
       predict_flux, predict_NDE,
       write_metadata_NN_training, write_data_NN_training, write_data_NN,
       write_metadata_NDE_training, write_data_NDE_training,
       train_NDE, train_NN,
       oceananigans_modified_pacanowski_philander_nn,
       solve_NDE_mutating, solve_oceananigans_modified_pacanowski_philander_nn,
       NDE_profile, NDE_profile_oceananigans, NDE_profile_unscaled, NDE_profile_mutating,
       animate_NN, animate_profile, animate_flux, animate_profile_flux, animate_profiles, animate_local_richardson_profile,
       animate_profiles_fluxes, animate_training_data_profiles_fluxes, animate_profiles_fluxes_comparison, 
       animate_training_results, animate_training_results_oceananigans,
       local_richardson,
       smoothing_filter,
       loss, loss_gradient,
       optimise_modified_pacanowski_philander,
       extract_NN, extract_parameters_modified_pacanowski_philander_optimisation,
       directories

using Flux, Plots
using Oceananigans.Grids: Center, Face
using Oceananigans: OceananigansLogger
using Oceananigans
using OceanParameterizations
import OceanTurb
using JLD2
using FileIO
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using GalacticOptim
using Statistics
using Random
using Logging
using CairoMakie
using Printf
using CUDA
using LinearAlgebra

mse(x::Tuple{Array{Float64,2}, Array{Float64,2}}) = Flux.mse(x[1], x[2])
mse(x::Tuple{Array{Float32,2}, Array{Float64,2}}) = Flux.mse(Float64.(x[1]), x[2])
mse(x::Tuple{Array{Float32,2}, Array{Float32,2}}) = Flux.mse(x[1], x[2])

include("lesbrary_data.jl")
include("data_containers.jl")
include("NDE_training.jl")
include("NN_training.jl")
include("animation.jl")
include("data_writing.jl")
include("filtering_operators.jl")
include("diffusivity_parameter_optimisation.jl")
include("data_extraction.jl")
include("NDE_oceananigans.jl")
include("training_postprocessing.jl")
include("k_profile_parameterization.jl")

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end