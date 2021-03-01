using Flux
using OceanParameterizations
using WindMixing

train_files = ["-1e-3"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")

# FILE_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "testNN.jld2")

FILE_PATH_uw = joinpath(OUTPUT_PATH, "uw_NN_training_1sim_-1e-3_small.jld2")
FILE_PATH_vw = joinpath(OUTPUT_PATH, "vw_NN_training_1sim_-1e-3_small.jld2")
FILE_PATH_wT = joinpath(OUTPUT_PATH, "wT_NN_training_1sim_-1e-3_small.jld2")

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

N_inputs = 96
hidden_units = 100
N_outputs = 31
uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))

train_optimizers = [ADAM(0.01), Descent()]
train_epochs = [50,100]

write_metadata_NN_training(FILE_PATH_uw, train_files, train_epochs, train_optimizers, uw_NN, "uw")
uw_weights = train_NN(uw_NN, 𝒟train.uvT_scaled, 𝒟train.uw.scaled, train_optimizers, train_epochs, FILE_PATH_uw, "uw")

write_metadata_NN_training(FILE_PATH_vw, train_files, train_epochs, train_optimizers, vw_NN, "vw")
uw_weights = train_NN(vw_NN, 𝒟train.uvT_scaled, 𝒟train.vw.scaled, train_optimizers, train_epochs, FILE_PATH_vw, "vw")

write_metadata_NN_training(FILE_PATH_wT, train_files, train_epochs, train_optimizers, wT_NN, "wT")
uw_weights = train_NN(wT_NN, 𝒟train.uvT_scaled, 𝒟train.wT.scaled, train_optimizers, train_epochs, FILE_PATH_wT, "wT")



# file = jldopen(FILE_PATH, "r")
# close(file)