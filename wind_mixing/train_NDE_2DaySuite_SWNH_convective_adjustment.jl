using WindMixing
using BSON
using OceanParameterizations
using Flux
using OrdinaryDiffEq, DiffEqSensitivity


train_files = ["strong_wind"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "Output")
FILE_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "test.jld2")

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

uw_NN_model = BSON.load(joinpath(PATH, "NDEs", "uw_NDE_SWNH_100.bson"))[:neural_network]
vw_NN_model = BSON.load(joinpath(PATH, "NDEs", "vw_NDE_SWNH_100.bson"))[:neural_network]
wT_NN_model = BSON.load(joinpath(PATH, "NDEs", "wT_NDE_SWNH_100.bson"))[:neural_network]

train_epochs = [100]
train_tranges = [1:5:100]
opts = [ADAM(0.01)]

write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, opts, uw_NN_model, vw_NN_model, wT_NN_model)


train_NDE_convective_adjustment(uw_NN_model, vw_NN_model, wT_NN_model, 𝒟train, 1:5:100, ROCK4(), [ADAM(0.01)], 100, OUTPUT_PATH, "weights_SWNH_convective_adjustment_100")
