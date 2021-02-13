using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random

# Training data
train_files = ["-1e-3"]

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")

# OUTPUT_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(OUTPUT_PATH, "NDE_training_modified_pacalowski_philander_1sim_-1e-3.jld2")

@assert !isfile(FILE_PATH)

# FILE_PATH_uw = joinpath(OUTPUT_PATH, "uw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")
# FILE_PATH_vw = joinpath(OUTPUT_PATH, "vw_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")
# FILE_PATH_wT = joinpath(OUTPUT_PATH, "wT_NN_training_2sim_-2.5e-4_-7.5e-4_large.jld2")

# uw_file = load(FILE_PATH_uw, "training_data/neural_network")
# vw_file = load(FILE_PATH_vw, "training_data/neural_network")
# wT_file = load(FILE_PATH_wT, "training_data/neural_network")

FILE_PATH_NN = joinpath(PATH, "extracted_training_output", "NDE_training_strong_convective_adjustment_1sim_-1e-3_2_extracted.jld2")
@assert isfile(FILE_PATH_NN)

# uw_file = jldopen(FILE_PATH_uw, "r")
# vw_file = jldopen(FILE_PATH_vw, "r")
# wT_file = jldopen(FILE_PATH_wT, "r")

file = jldopen(FILE_PATH_NN, "r")

uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

# train_epochs = [1 for i in 1:100]
# train_tranges = [1:rand(10:1:40):1153 for i in 1:length(train_epochs)]

train_epochs = [4, 4, 4, 4, 4, 4, 4, 4, 20]
train_tranges = [1:1:10, 1:1:20, 1:5:50, 1:5:100, 1:10:200, 1:10:300, 1:25:500, 1:30:700, 1:20:1153]

train_optimizers = [[ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM()]]
timestepper = ROCK4()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
        # uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 10f0, 5)
        uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, 1, 5, modified_pacalowski_philander=true)
    end

end

train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
