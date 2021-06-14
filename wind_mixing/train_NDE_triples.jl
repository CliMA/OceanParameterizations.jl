using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random
using GalacticOptim
using LinearAlgebra

BLAS.set_num_threads(1)

# Training data
# train_files = [
#     "wind_-1e-3_heating_-4e-8",
#     "wind_-5e-4_cooling_4e-8",
#     ]

train_files = [
    "-1e-3"
]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

PATH = pwd()

OUTPUT_PATH = joinpath(PATH, "training_output")
OUTPUT_PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output"

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output")

# FILE_PATH = joinpath(OUTPUT_PATH, "NDE_training_mpp_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4.jld2")
# @assert !isfile(FILE_PATH)


# FILE_PATH_uw = joinpath(PATH, "extracted_training_output", "uw_NN_training_1sim_-1e-3_extracted.jld2")
# FILE_PATH_vw = joinpath(PATH, "extracted_training_output", "vw_NN_training_1sim_-1e-3_extracted.jld2")
# FILE_PATH_wT = joinpath(PATH, "extracted_training_output", "wT_NN_training_1sim_-1e-3_extracted.jld2")

# uw_file = jldopen(FILE_PATH_uw, "r")
# vw_file = jldopen(FILE_PATH_vw, "r")
# wT_file = jldopen(FILE_PATH_wT, "r")

# uw_NN = uw_file["neural_network"]
# vw_NN = vw_file["neural_network"]
# wT_NN = wT_file["neural_network"]

N_inputs = 96
hidden_units = 400
N_outputs = 31

weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

uw_NN = re(weights ./ 1f5)
vw_NN = re(weights ./ 1f5)
wT_NN = re(weights ./ 1f5)

task_id = parse(Int,ARGS[1]) + 1
num_tasks = parse(Int,ARGS[2])
task_id = 1
FILE_NAME = ["NDE_training_mpp_2sim_windcooling_MS_windheating_SS_diffusivity_1e-1_Ri_1e-1_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4",
             "NDE_training_mpp_2sim_windcooling_MS_windheating_SS_diffusivity_1e-1_Ri_1e-1_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4",
             "NDE_training_mpp_2sim_windcooling_MS_windheating_SS_diffusivity_1e-1_Ri_1e-1_divide1f5_gradient_smallNN_scale_1.5e-2_rate_1e-4",
             "NDE_training_mpp_2sim_windcooling_MS_windheating_SS_diffusivity_1e-1_Ri_1e-1_divide1f5_gradient_smallNN_scale_2e-2_rate_1e-4",
              ][task_id]

FILE_PATH = joinpath(OUTPUT_PATH, "$(FILE_NAME).jld2")
@assert !isfile(FILE_PATH)

EXTRACTED_FILE_NAME = "$(FILE_NAME)_extracted"
EXTRACTED_FILE_PATH = joinpath(EXTRACTED_OUTPUT_PATH, "$EXTRACTED_FILE_NAME.jld2")

# FILE_NAME_NN = ["NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4_extracted.jld2",
#                 "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_2e-4_extracted.jld2",
#                 "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4_extracted.jld2",
#                 "NDE_training_mpp_5sim_-1e-3_-9e-4_-8e-4_-7e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4_extracted.jld2"
#                 ][task_id]

# FILE_NAME_NN = "NDE_training_mpp_9sim_windcooling_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4_extracted.jld2"
# FILE_PATH_NN = joinpath(EXTRACTED_OUTPUT_PATH, FILE_NAME_NN)

# @assert isfile(FILE_PATH_NN)
# file = jldopen(FILE_PATH_NN, "r")

# uw_NN = file["neural_network/uw"]
# vw_NN = file["neural_network/vw"]
# wT_NN = file["neural_network/wT"]

# close(file)


gradient_scaling = [5f-3, 1f-2, 1.5f-2, 2f-2][task_id]
train_parameters = Dict("ν₀" => 1f-4, "ν₋" => 0.1f0, "Riᶜ" => 0.25f0, "ΔRi" => 1f-1, "Pr" => 1f0, "κ" => 10f0,
                        "modified_pacanowski_philander" => true, "convective_adjustment" => false,
                        "smooth_profile" => false, "smooth_NN" => false, "smooth_Ri" => false, "train_gradient" => true,
                        "zero_weights" => true, "unscaled" => false, "gradient_scaling" => gradient_scaling)

# train_epochs = [1]
# train_tranges = [1:9:1153]
# train_iterations = [600]
# # train_optimizers = [[[ADAM(1e-4)]], [[ADAM(2e-4)]], [[ADAM(1e-4)]], [[ADAM(2e-4)]], [[ADAM(1e-4)]], [[ADAM(2e-4)]], [[ADAM(1e-4)]], [[ADAM(2e-4)]]][task_id]
# train_optimizers = [[ADAM(1e-4)]]

train_epochs = [1]
train_tranges = [1:35:200]
train_iterations = [3]
train_optimizers = [[ADAM(2e-4)]]

# train_tranges = [1:10:100, 1:10:200, 1:20:500, 1:30:700, 1:30:800, 1:30:900, 1:35:1153]
# train_epochs = [1 for i in 1:length(train_tranges)]
# train_iterations = [50, 50, 100, 30, 20, 50, 150]
# train_optimizers = [[ADAM(0.1), ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01)], [ADAM(0.01), ADAM(0.001), ADAM(5e-4), ADAM(2e-4)]]

# train_tranges = [1:10:100, 1:10:200, 1:20:500, 1:20:800, 1:35:1153]
# train_epochs = [1 for i in 1:length(train_tranges)]
# train_iterations = [30, 30, 50, 30, 200]
# train_optimizers = [[[ADAM(0.01)] for i in 1:6]; [[ADAM(0.01), ADAM(1e-3)]]]
# # train_optimizers = [[ADAM(1e-5)] for i in 1:6]


timestepper = Rosenbrock23()

# train_optimizers = [[ADAM(2e-4), ADAM(1e-4), ADAM(5e-5), RMSProp(1e-4)]]
# train_optimizers=[[ADAM(5e-4)]]

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

# uw_NN_res, vw_NN_res, wT_NN_res = train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper, train_parameters["unscaled"])

uw_NN_res, vw_NN_res, wT_NN_res =  train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, 𝒟train, timestepper, train_parameters["unscaled"])

extract_NN(FILE_PATH, EXTRACTED_FILE_PATH, "NDE")

test_files = train_files

animate_training_results(test_files, FILE_NAME, trange=1:1:1153)

OCEANANIGANS_OUTPUT_DIR = joinpath(pwd(), "NDE_output_oceananigans", FILE_NAME)
animate_training_results_oceananigans(test_files, 60, FILE_NAME, OCEANANIGANS_OUTPUT_DIR)