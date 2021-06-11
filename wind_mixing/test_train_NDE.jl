using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random
using GalacticOptim
using LinearAlgebra

train_files = [
        "-1e-3",       
]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
# 
PATH = pwd()

OUTPUT_PATH = joinpath(PATH, "training_output")
# OUTPUT_PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output"

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output")

FILE_NAME = "test"
FILE_PATH = joinpath(OUTPUT_PATH, "$(FILE_NAME).jld2")
EXTRACTED_FILE_PATH = joinpath(EXTRACTED_OUTPUT_PATH, "$(FILE_NAME)_extracted.jld2")
@assert !isfile(FILE_PATH)

N_inputs = 96
hidden_units = 400
N_outputs = 31

weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

uw_NN = re(weights ./ 1f5)
vw_NN = re(weights ./ 1f5)
wT_NN = re(weights ./ 1f5)

gradient_scaling = 5f-3
train_parameters = Dict("ν₀" => 1f-4, "ν₋" => 0.1f0, "Riᶜ" => 0.25f0, "ΔRi" => 1f-1, "Pr" => 1f0, "κ" => 10f0,
                        "modified_pacanowski_philander" => true, "convective_adjustment" => false,
                        "smooth_profile" => false, "smooth_NN" => false, "smooth_Ri" => false, "train_gradient" => true,
                        "zero_weights" => true, "unscaled" => false, "gradient_scaling" => gradient_scaling)

train_epochs = [1]
train_tranges = [1:20:200]
train_iterations = [3]
train_optimizers = [[ADAM(2e-4)]]


timestepper = Rosenbrock23()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper, unscaled)
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, uw_NN, vw_NN, wT_NN)
    if unscaled
        for i in 1:length(train_epochs)
            @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
            # uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 10f0, 5)
            if train_parameters["modified_pacanowski_philander"]
                uw_NN, vw_NN, wT_NN = train_NDE_unscaled(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=length(train_files), maxiters=train_iterations[i], 
                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], convective_adjustment=train_parameters["convective_adjustment"],
                ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                κ=train_parameters["κ"],
                smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], train_gradient=train_parameters["train_gradient"],
                gradient_scaling=train_parameters["gradient_scaling"])
            else
                uw_NN, vw_NN, wT_NN = train_NDE_unscaled(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=length(train_files), maxiters=train_iterations[i], 
                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], convective_adjustment=train_parameters["convective_adjustment"],
                κ=train_parameters["κ"],
                smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], train_gradient=train_parameters["train_gradient"],
                gradient_scaling=train_parameters["gradient_scaling"])
            end
        end
    else
        for i in 1:length(train_epochs)
            @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
            # uw_NN, vw_NN, wT_NN = train_NDE_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, 1, 1, 10f0, 5)
            if train_parameters["modified_pacanowski_philander"]
                uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=length(train_files), maxiters=train_iterations[i], 
                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], convective_adjustment=train_parameters["convective_adjustment"],
                ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                κ=train_parameters["κ"],
                smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], train_gradient=train_parameters["train_gradient"],
                zero_weights = train_parameters["zero_weights"],
                gradient_scaling=train_parameters["gradient_scaling"])
            else
                uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, 𝒟train, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, n_simulations=length(train_files), maxiters=train_iterations[i], 
                modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], convective_adjustment=train_parameters["convective_adjustment"],
                κ=train_parameters["κ"],
                smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], train_gradient=train_parameters["train_gradient"],
                zero_weights = train_parameters["zero_weights"],
                gradient_scaling=train_parameters["gradient_scaling"])
            end
        end
    end
    return uw_NN, vw_NN, wT_NN
end

uw_NN_res, vw_NN_res, wT_NN_res = train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper, train_parameters["unscaled"])