using Flux
using OceanParameterizations
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using WindMixing

# data in which the neural network is trained on
train_files = ["strong_wind"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "Output")

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

uvT = 𝒟train.uvT_scaled
uw = 𝒟train.uw.scaled
vw = 𝒟train.vw.scaled
wT = 𝒟train.wT.scaled

N_inputs = 96
hidden_units = 400
N_outputs = 31
uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))

_, re_uw = Flux.destructure(uw_NN)
_, re_vw = Flux.destructure(vw_NN)
_, re_wT = Flux.destructure(wT_NN)

optimizers_NN = [ADAM(0.01)]
epochs_NN = 10

uw_weights = train_NN(uw_NN, uvT, uw, optimizers_NN, epochs_NN, OUTPUT_PATH, "uw_weights_large_NN")
vw_weights = train_NN(vw_NN, uvT, vw, optimizers_NN, epochs_NN, OUTPUT_PATH, "vw_weights_large_NN")
wT_weights = train_NN(wT_NN, uvT, wT, optimizers_NN, epochs_NN, OUTPUT_PATH, "wT_weights_large_NN")

uw_NN = re_uw(uw_weights)
vw_NN = re_vw(vw_weights)
wT_NN = re_wT(wT_weights)

train_tranges_NDE = [1:5:50, 1:5:100]
optimizers_NDE = [ADAM(0.01)]
epochs_NDE = 100

train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, 1:5:100, ROCK4(), optimizers_NDE, epochs_NDE, OUTPUT_PATH, "weights_SWNH_convective_adjustment_100_large")


