using Flux
using WindMixing
using BSON
using OceanParameterizations
using OrdinaryDiffEq

# Training data
train_files = ["-1e-3"]

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
PATH=pwd()
FILE_PATH = joinpath(PATH, "training_output", "NDE_training_convective_adjustment_1sim_-1e-3_large.jld2")

uw_NN = BSON.load(joinpath(PATH, "NDEs", "uw_NN_large.bson"))[:neural_network]
vw_NN = BSON.load(joinpath(PATH, "NDEs", "vw_NN_large.bson"))[:neural_network]
wT_NN = BSON.load(joinpath(PATH, "NDEs", "wT_NN_large.bson"))[:neural_network]


train_epochs = [1, 1, 1]
train_tranges = [1:5:50, 1:5:100, 1:10:200]
train_optimizers = [[ADAM(0.01)] for i in 1:3]
timestepper = ROCK4()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, 1)
    end

end

train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)