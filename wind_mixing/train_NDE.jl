using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random
using GalacticOptim
using LinearAlgebra

BLAS.set_num_threads(32)

# Training data
train_files = ["-1e-3"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
# 
PATH = pwd()

OUTPUT_PATH = joinpath(PATH, "training_output")
# OUTPUT_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"

FILE_PATH = joinpath(OUTPUT_PATH, "NDE_training_1sim_-1e-3_gradient.jld2")
@assert !isfile(FILE_PATH)

FILE_PATH_uw = joinpath(PATH, "extracted_training_output", "uw_NN_training_1sim_-1e-3_extracted.jld2")
FILE_PATH_vw = joinpath(PATH, "extracted_training_output", "vw_NN_training_1sim_-1e-3_extracted.jld2")
FILE_PATH_wT = joinpath(PATH, "extracted_training_output", "wT_NN_training_1sim_-1e-3_extracted.jld2")

uw_file = jldopen(FILE_PATH_uw, "r")
vw_file = jldopen(FILE_PATH_vw, "r")
wT_file = jldopen(FILE_PATH_wT, "r")

uw_NN = uw_file["neural_network"]
vw_NN = vw_file["neural_network"]
wT_NN = wT_file["neural_network"]

# FILE_PATH_NN = joinpath(PATH, "extracted_training_output", 
#                         "NDE_training_2sim_-1e-3_-8e-4_smooth_NN_3_extracted.jld2")

# @assert isfile(FILE_PATH_NN)
# file = jldopen(FILE_PATH_NN, "r")

# uw_NN = file["neural_network/uw"]
# vw_NN = file["neural_network/vw"]
# wT_NN = file["neural_network/wT"]

train_parameters = Dict("ν₀" => 1f-4, "ν₋" => 0.1f0, "Riᶜ" => 0.25f0, "ΔRi" => 1f0, "Pr" => 1f0, "κ" => 10f0,
                        "modified_pacanowski_philander" => false, "convective_adjustment" => false,
                        "smooth_profile" => false, "smooth_NN" => false, "smooth_Ri" => false, "train_gradient" => true)

# train_epochs = [1]
# train_tranges = [1:31:1153]
# train_iterations = [500]
# train_optimizers = [[ADAM(1e-3)]]

# train_epochs = [1]
# train_tranges = [1:20:100]
# train_iterations = [5]
# train_optimizers = [[ADAM(0.01)]]

# train_tranges = [1:10:100, 1:10:200, 1:20:500, 1:30:700, 1:30:800, 1:30:900, 1:35:1153]
# train_epochs = [1 for i in 1:length(train_tranges)]
# train_iterations = [50, 50, 100, 30, 20, 50, 150]
# train_optimizers = [[ADAM(0.1), ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01), ADAM(0.001), ADAM(5e-4), ADAM(2e-4)]]

train_tranges = [1:10:100, 1:10:200, 1:20:500, 1:20:800, 1:35:1153]
train_epochs = [1 for i in 1:length(train_tranges)]
train_iterations = [30, 30, 50, 30, 200]
train_optimizers = [[[ADAM(0.01)] for i in 1:6]; [[ADAM(1e-3)]]]

timestepper = ROCK4()

# train_optimizers = [[ADAM(2e-4), ADAM(1e-4), ADAM(5e-5), RMSProp(1e-4)]]
# train_optimizers=[[ADAM(5e-4)]]

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, uw_NN, vw_NN, wT_NN)

    for i in 1:length(train_epochs)
        @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
        # uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 10f0, 5)
        if train_parameters["modified_pacanowski_philander"]
            uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=length(train_files), maxiters=train_iterations[i], 
            modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], convective_adjustment=train_parameters["convective_adjustment"],
            ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
            κ=train_parameters["κ"],
            smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], train_gradient=train_parameters["train_gradient"])
        else
            uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=length(train_files), maxiters=train_iterations[i], 
            modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], convective_adjustment=train_parameters["convective_adjustment"],
            κ=train_parameters["κ"],
            smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], train_gradient=train_parameters["train_gradient"])
        end
    end
end

train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper)
