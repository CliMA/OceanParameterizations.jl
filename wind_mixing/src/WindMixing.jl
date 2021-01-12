module WindMixing

export data, read_les_output,
       animate_prediction,
       mse, 
       predict_NDE, predict_NDE_convective_adjustment,
       train_NDE, train_NDE_convective_adjustment, train_NN, 
       NDE_profile, NDE_profile_convective_adjustment, animate_NN, animate_profile, animate_flux, animate_profile_flux, 
       write_metadata_NDE_training, write_data_NDE_training,
       write_metadata_NN_training, write_data_NN_training

using Flux, Plots
using Oceananigans.Grids: Cell, Face
using OceanParameterizations
using JLD2
using FileIO
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using GalacticOptim
using Statistics
using Random

mse(x::Tuple{Array{Float64,2}, Array{Float64,2}}) = Flux.mse(x[1], x[2])
mse(x::Tuple{Array{Float32,2}, Array{Float64,2}}) = Flux.mse(Float64.(x[1]), x[2])

include("lesbrary_data.jl")
include("data_containers.jl")
include("NDE_training.jl")
include("NN_training.jl")
include("animation.jl")
include("training_data.jl")

end
