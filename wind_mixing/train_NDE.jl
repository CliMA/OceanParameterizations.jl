using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random
using GalacticOptim

# Training data
train_files = ["-1e-3"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")

# OUTPUT_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(OUTPUT_PATH, "NDE_training_modified_pacalowski_philander_1sim_-1e-3_diffusivity_3e-1_Ri_1e-1_extracted.jld2")

@assert !isfile(FILE_PATH)

FILE_PATH_uw = joinpath(PATH, "extracted_training_output", "uw_NN_training_1sim_-1e-3_extracted.jld2")
FILE_PATH_vw = joinpath(PATH, "extracted_training_output", "vw_NN_training_1sim_-1e-3_extracted.jld2")
FILE_PATH_wT = joinpath(PATH, "extracted_training_output", "wT_NN_training_1sim_-1e-3_extracted.jld2")

# FILE_PATH_NN = joinpath(PATH, "extracted_training_output", "NDE_training_modified_pacalowski_philander_1sim_-1e-3_higher_diffusivity_extracted.jld2")

@assert isfile(FILE_PATH_NN)
file = jldopen(FILE_PATH_NN, "r")

# uw_NN = file["neural_network/uw"]
# vw_NN = file["neural_network/vw"]
# wT_NN = file["neural_network/wT"]

uw_file = jldopen(FILE_PATH_uw, "r")
vw_file = jldopen(FILE_PATH_vw, "r")
wT_file = jldopen(FILE_PATH_wT, "r")

uw_NN = uw_file["neural_network"]
vw_NN = vw_file["neural_network"]
wT_NN = wT_file["neural_network"]

# train_epochs = [1]
# train_tranges = [1:20:1153]
# train_iterations = [5]


train_tranges = [1:10:100, 1:10:200, 1:20:500, 1:20:700, 1:20:800, 1:20:900, 1:20:1153]
train_epochs = [1 for i in 1:length(train_tranges)]
train_iterations = [20, 20, 30, 30, 40, 50, 50]

train_optimizers = [[[ADAM(0.01)] for i in 1:6]; [[ADAM(0.01), ADAM(1e-3), ADAM(5e-4), ADAM(2e-4), ADAM(1e-4), RMSProp(1e-4)]]]
timestepper = ROCK4()

# train_optimizers = [[ADAM(2e-4), ADAM(1e-4), ADAM(5e-5), RMSProp(1e-4)]]
# train_optimizers=[[ADAM(5e-4)]]

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
        # uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 10f0, 5)
        uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=1, maxiters=train_iterations[i], modified_pacalowski_philander=true, ν₀=1f-4, ν₋=3f-1, ΔRi=1f-1, convective_adjustment=false)
        # uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 5, modified_pacalowski_philander=true)
    end

end

train(FILE_PATH, train_files, train_epochs, train_tranges, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
