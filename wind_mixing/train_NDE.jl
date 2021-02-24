using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random

# Training data
train_files = ["-1e-3"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")

OUTPUT_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(OUTPUT_PATH, "NDE_training_modified_pacalowski_philander_1sim_-1e-3_higher_diffusivity.jld2")

@assert !isfile(FILE_PATH)

FILE_PATH_uw = joinpath(PATH, "extracted_training_output", "uw_NN_training_1sim_-1e-3_extracted.jld2")
FILE_PATH_vw = joinpath(PATH, "extracted_training_output", "vw_NN_training_1sim_-1e-3_extracted.jld2")
FILE_PATH_wT = joinpath(PATH, "extracted_training_output", "wT_NN_training_1sim_-1e-3_extracted.jld2")

# FILE_PATH_NN = joinpath(PATH, "extracted_training_output", "NDE_training_convective_adjustment_1sim_-1e-3_smallADAM_extracted.jld2")
# @assert isfile(FILE_PATH_NN)
# file = jldopen(FILE_PATH_NN, "r")

# uw_NN = file["neural_network/uw"]
# vw_NN = file["neural_network/vw"]
# wT_NN = file["neural_network/wT"]

uw_file = jldopen(FILE_PATH_uw, "r")
vw_file = jldopen(FILE_PATH_vw, "r")
wT_file = jldopen(FILE_PATH_wT, "r")

uw_NN = uw_file["neural_network"]
vw_NN = vw_file["neural_network"]
wT_NN = wT_file["neural_network"]


# train_epochs = [1 for i in 1:100]
# train_tranges = [1:rand(10:1:40):1153 for i in 1:length(train_epochs)]

train_tranges = [1:10:100, 1:10:200, 1:20:500, 1:20:700, 1:20:800, 1:20:900, 1:20:1153]
train_epochs = [[6 for i in 1:6]; [200]]

train_optimizers = [[[ADAM(0.01)] for i in 1:6]; [[ADAM()]]]
timestepper = ROCK4()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
        # uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 10f0, 5)
        uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=1, maxiters=5, modified_pacalowski_philander=true, ν₋=1f0, ΔRi=1f-1)
        # uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 5, modified_pacalowski_philander=true)
    end

end

train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
