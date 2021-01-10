using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq

# Training data
train_files = ["-2.5e-4", "-7.5e-4"]

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")

FILE_PATH = joinpath(OUTPUT_PATH, "NDE_training_convective_adjustment_2sim_-2.5e-4_-7.5e-4_large.jld2")

FILE_PATH_uw = joinpath(OUTPUT_PATH, "uw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")
FILE_PATH_vw = joinpath(OUTPUT_PATH, "vw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")
FILE_PATH_wT = joinpath(OUTPUT_PATH, "wT_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")

# uw_file = load(FILE_PATH_uw, "training_data/neural_network")
# vw_file = load(FILE_PATH_vw, "training_data/neural_network")
# wT_file = load(FILE_PATH_wT, "training_data/neural_network")

uw_file = jldopen(FILE_PATH_uw, "r")
vw_file = jldopen(FILE_PATH_vw, "r")
wT_file = jldopen(FILE_PATH_wT, "r")

uw_NN = uw_file["training_data/neural_network"]["$(length(keys(uw_file["training_data/neural_network"])))"]
vw_NN = vw_file["training_data/neural_network"]["$(length(keys(vw_file["training_data/neural_network"])))"]
wT_NN = wT_file["training_data/neural_network"]["$(length(keys(wT_file["training_data/neural_network"])))"]

train_epochs = [20, 20, 20, 20, 40]
train_tranges = [1:5:50, 1:5:100, 1:10:200, 1:20:400, 1:20:500]
train_optimizers = [[ADAM(0.01)] for i in 1:length(train_epochs)]
timestepper = ROCK4()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, 1)
    end

end

train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)