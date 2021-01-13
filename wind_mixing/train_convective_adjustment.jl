using Flux
using OceanParameterizations
using WindMixing
using OrdinaryDiffEq, DiffEqSensitivity
using JLD2
using FileIO

train_files = ["-1e-3", "-5e-4"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")

# FILE_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "testNN.jld2")

FILE_PATH_uw = joinpath(OUTPUT_PATH, "uw_NN_training_2sim_-5e-4_-1e-3_large.jld2")
FILE_PATH_vw = joinpath(OUTPUT_PATH, "vw_NN_training_2sim_-5e-4_-1e-3_large.jld2")
FILE_PATH_wT = joinpath(OUTPUT_PATH, "wT_NN_training_2sim_-5e-4_-1e-3_large.jld2")

# FILE_PATH_uw = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "uw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")
# FILE_PATH_vw = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "vw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")
# FILE_PATH_wT = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "wT_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

N_inputs = 96
hidden_units = 400
N_outputs = 31
uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))

train_optimizers = [ADAM(0.01), Descent()]
train_epochs = [10,10]

write_metadata_NN_training(FILE_PATH_uw, train_files, train_epochs, train_optimizers, uw_NN, "uw")
uw_weights = train_NN(uw_NN, 𝒟train.uvT_scaled, 𝒟train.uw.scaled, train_optimizers, train_epochs, FILE_PATH_uw, "uw")

write_metadata_NN_training(FILE_PATH_vw, train_files, train_epochs, train_optimizers, vw_NN, "vw")
vw_weights = train_NN(vw_NN, 𝒟train.uvT_scaled, 𝒟train.vw.scaled, train_optimizers, train_epochs, FILE_PATH_vw, "vw")

write_metadata_NN_training(FILE_PATH_wT, train_files, train_epochs, train_optimizers, wT_NN, "wT")
wT_weights = train_NN(wT_NN, 𝒟train.uvT_scaled, 𝒟train.wT.scaled, train_optimizers, train_epochs, FILE_PATH_wT, "wT")


uw_NN = Flux.destructure(uw_NN)[2](uw_weights)
vw_NN = Flux.destructure(vw_NN)[2](vw_weights)
wT_NN = Flux.destructure(wT_NN)[2](wT_weights)

FILE_PATH_NDE = joinpath(OUTPUT_PATH, "NDE_training_convective_adjustment_2sim_-5e-4_-1e-3_large.jld2")

train_epochs = [1, 1, 1, 1, 1, 1, 1, 1, 10]
train_tranges = [1:5:50, 1:5:100, 1:10:200, 1:20:400, 1:20:500, 1:25:700, 1:25:900, 1:30:1000, 1:40:1153]
train_optimizers = [[ADAM(0.01)] for i in 1:length(train_epochs)]
timestepper = ROCK4()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, 1)
    end

end

train(FILE_PATH_NDE, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)