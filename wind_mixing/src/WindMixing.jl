module WindMixing

export data, read_les_output,
       animate_prediction,
       mse, 
       prepare_parameters_NDE_training_unscaled,
       predict_flux, predict_NDE,
       train_NDE, train_NN,
       NDE_profile, animate_NN, animate_profile, animate_flux, animate_profile_flux, animate_profiles, animate_local_richardson_profile,
       NDE_profile_oceananigans, NDE_profile_unscaled, solve_NDE_mutating,
       animate_profiles_fluxes, animate_training_data_profiles_fluxes, animate_profiles_fluxes_comparison, animate_training_results,
       write_metadata_NDE_training, write_data_NDE_training,
       write_metadata_NN_training, write_data_NN_training, write_data_NN,
       local_richardson,
       smoothing_filter,
       loss, loss_gradient,
       optimise_modified_pacanowski_philander,
       extract_NN, extract_parameters_modified_pacanowski_philander_optimisation

using Flux, Plots
using Oceananigans.Grids: Center, Face
using Oceananigans: OceananigansLogger
# using Oceananigans
using OceanParameterizations
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

function __init__()
    Logging.global_logger(OceananigansLogger())
end

end
