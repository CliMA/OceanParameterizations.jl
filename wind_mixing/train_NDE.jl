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

# train_files = [
#     # "wind_-5e-4_cooling_4e-8", 
#     # "wind_-1e-3_cooling_4e-8", 
#     # "wind_-2e-4_cooling_1e-8", 
#     # "wind_-1e-3_cooling_2e-8", 
#     # "wind_-5e-4_cooling_1e-8", 
#     # "wind_-2e-4_cooling_5e-8", 
#     # "wind_-5e-4_cooling_3e-8", 
#     "wind_-2e-4_cooling_3e-8", 
#     # "wind_-1e-3_cooling_3e-8", 
#     # "wind_-1e-3_heating_-4e-8",
#     # "wind_-1e-3_heating_-1e-8",
#     # "wind_-1e-3_heating_-3e-8",
#     # "wind_-5e-4_heating_-5e-8",
#     # "wind_-5e-4_heating_-3e-8",
#     # "wind_-5e-4_heating_-1e-8",
#     # "wind_-2e-4_heating_-5e-8",
#     # "wind_-2e-4_heating_-3e-8",
#     # "wind_-2e-4_heating_-1e-8",
# ]

train_files = ["wind_-5e-4_cooling_3e-8"]

 PATH = pwd()

OUTPUT_PATH = joinpath(PATH, "training_output")
OUTPUT_PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output"

VIDEO_PATH = joinpath(PATH, "Output")

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output")
EXTRACTED_OUTPUT_PATH = OUTPUT_PATH

FILE_NAME = "NDE_18sim_windcooling_windheating_18sim5paramsBFGS_divide1f5_gradient_smallNN_mish_scale_5e-3_rate_1e-4"
FILE_PATH = joinpath(OUTPUT_PATH, "$(FILE_NAME).jld2")

EXTRACTED_FILE_PATH = joinpath(EXTRACTED_OUTPUT_PATH, "$(FILE_NAME)_extracted.jld2")
# @assert !isfile(FILE_PATH)

# PARAMETERS_PATH = joinpath(EXTRACTED_OUTPUT_PATH, "parameter_optimisation_18sim_windcooling_windheating_5params_BFGS_extracted.jld2")

# parameters_file = jldopen(PARAMETERS_PATH)
# mpp_parameters = parameters_file["parameters"]
# close(parameters_file)

# ν₀_initial = 1f-4
# ν₋_initial = 1f-1
# ΔRi_initial = 1f-1
# Riᶜ_initial = 0.25f0
# Pr_initial = 1f0

# mpp_scalings = 1 ./ [ν₀_initial, ν₋_initial, ΔRi_initial, Riᶜ_initial, Pr_initial]

# ν₀, ν₋, ΔRi, Riᶜ, Pr = mpp_parameters ./ mpp_scalings

ν₀ = 1f-4
ν₋ = 1f-1
ΔRi = 1f-1
Riᶜ = 0.25f0
Pr = 1f0

# FILE_PATH_uw = joinpath(PATH, "extracted_training_output", "uw_NN_training_1sim_-1e-3_extracted.jld2")
# FILE_PATH_vw = joinpath(PATH, "extracted_training_output", "vw_NN_training_1sim_-1e-3_extracted.jld2")
# FILE_PATH_wT = joinpath(PATH, "extracted_training_output", "wT_NN_training_1sim_-1e-3_extracted.jld2")

# uw_file = jldopen(FILE_PATH_uw, "r")
# vw_file = jldopen(FILE_PATH_vw, "r")
# wT_file = jldopen(FILE_PATH_wT, "r")

# uw_NN = uw_file["neural_network"]
# vw_NN = vw_file["neural_network"]
# wT_NN = wT_file["neural_network"]

# FILE_PATH_NN = joinpath(PATH, "extracted_training_output", 
#                         "NDE_training_mpp_8sim_wind_mixing_cooling_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4_extracted.jld2")

# @assert isfile(FILE_PATH_NN)
# file = jldopen(FILE_PATH_NN, "r")

# uw_NN = file["neural_network/uw"]
# vw_NN = file["neural_network/vw"]
# wT_NN = file["neural_network/wT"]

N_inputs = 96
hidden_units = 400
N_outputs = 31

# weights, re = Flux.destructure(Chain(Dense(N_inputs, 32, swish), Dense(32, 400, swish), Dense(400, N_outputs)))
weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, leakyrelu), Dense(hidden_units, N_outputs)))
# weights, re = Flux.destructure(Chain(Dense(N_inputs, 50, mish), Dense(50, 20, mish), Dense(20, 31)))

uw_NN = re(weights ./ 1f5)
vw_NN = re(weights ./ 1f5)
wT_NN = re(weights ./ 1f5)

gradient_scaling = 5f-3
training_fractions = (T=0.8f0, ∂T∂z=0.8f0, profile=0.5f0)

diurnal = occursin("diurnal", train_files[1])

train_parameters = Dict(
                               "ν₀" => ν₀, 
                               "ν₋" => ν₋, 
                              "ΔRi" => ΔRi, 
                              "Riᶜ" => Riᶜ, 
                               "Pr" => Pr, 
                                "κ" => 10f0,
    "modified_pacanowski_philander" => true, 
            "convective_adjustment" => false,
                   "smooth_profile" => false, 
                        "smooth_NN" => false, 
                        "smooth_Ri" => false, 
                   "train_gradient" => true,
                     "zero_weights" => true, 
                 "gradient_scaling" => gradient_scaling, 
               "training_fractions" => training_fractions,
                          "diurnal" => diurnal,
    )

# train_epochs = [1]
# train_tranges = [1:9:1153]
# train_iterations = [600]
# train_optimizers = [[ADAM(1e-4)]]

train_epochs = [1]
train_tranges = [1:20:200]
train_iterations = [3]
train_optimizers = [[ADAM(3e-4)]]

timestepper = Rosenbrock23()

function train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, uw_NN, vw_NN, wT_NN, timestepper)
    @info "Writing metadata"
    write_metadata_NDE_training(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, uw_NN, vw_NN, wT_NN)
    for i in 1:length(train_epochs)
        @info "iteration $i/$(length(train_epochs)), time range $(train_tranges[i])"
        if train_parameters["modified_pacanowski_philander"]
            uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, train_files, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, 
                                                         maxiters = train_iterations[i], 
                                    modified_pacanowski_philander = train_parameters["modified_pacanowski_philander"], 
                                            convective_adjustment = train_parameters["convective_adjustment"],
                                                               ν₀ = train_parameters["ν₀"], 
                                                               ν₋ = train_parameters["ν₋"], 
                                                              ΔRi = train_parameters["ΔRi"], 
                                                              Riᶜ = train_parameters["Riᶜ"], 
                                                               Pr = train_parameters["Pr"],
                                                                κ = train_parameters["κ"],
                                                   smooth_profile = train_parameters["smooth_profile"], 
                                                        smooth_NN = train_parameters["smooth_NN"], 
                                                        smooth_Ri = train_parameters["smooth_Ri"], 
                                                   train_gradient = train_parameters["train_gradient"],
                                                     zero_weights = train_parameters["zero_weights"],
                                                #  gradient_scaling = train_parameters["gradient_scaling"],
                                               training_fractions = train_parameters["training_fractions"],
                                                          diurnal = train_parameters["diurnal"],
                                    )
        else
            uw_NN, vw_NN, wT_NN = train_NDE(uw_NN, vw_NN, wT_NN, train_files, train_tranges[i], timestepper, train_optimizers[i], train_epochs[i], FILE_PATH, i, 
                                                         maxiters = train_iterations[i], 
                                    modified_pacanowski_philander = train_parameters["modified_pacanowski_philander"], 
                                            convective_adjustment = train_parameters["convective_adjustment"],
                                                                κ = train_parameters["κ"],
                                                   smooth_profile = train_parameters["smooth_profile"],
                                                        smooth_NN = train_parameters["smooth_NN"], 
                                                        smooth_Ri = train_parameters["smooth_Ri"],
                                                   train_gradient = train_parameters["train_gradient"],
                                                     zero_weights = train_parameters["zero_weights"],
                                                #  gradient_scaling = train_parameters["gradient_scaling"],
                                               training_fractions = train_parameters["training_fractions"],
                                                          diurnal = train_parameters["diurnal"],
                                                )
        end
    end
    return uw_NN, vw_NN, wT_NN
end

uw_NN_res, vw_NN_res, wT_NN_res = train(FILE_PATH, train_files, train_epochs, train_tranges, train_parameters, train_optimizers, train_iterations, 
                                    uw_NN, vw_NN, wT_NN, timestepper)

# extract_NN(FILE_PATH, EXTRACTED_FILE_PATH, "NDE")

# test_files = [
#     # "wind_-5e-4_cooling_4e-8", 
#     # "wind_-1e-3_cooling_4e-8", 
#     # "wind_-2e-4_cooling_1e-8", 
#     # "wind_-1e-3_cooling_2e-8", 
#     # "wind_-5e-4_cooling_1e-8", 
#     # "wind_-2e-4_cooling_5e-8", 
#     # "wind_-5e-4_cooling_3e-8", 
#     "wind_-2e-4_cooling_3e-8", 
#     # "wind_-1e-3_cooling_3e-8", 
#     # "wind_-1e-3_heating_-4e-8",
#     # "wind_-1e-3_heating_-1e-8",
#     # "wind_-1e-3_heating_-3e-8",
#     # "wind_-5e-4_heating_-5e-8",
#     # "wind_-5e-4_heating_-3e-8",
#     # "wind_-5e-4_heating_-1e-8",
#     # "wind_-2e-4_heating_-5e-8",
#     # "wind_-2e-4_heating_-3e-8",
#     # "wind_-2e-4_heating_-1e-8",
# ]

# animate_training_results(test_files, DATA_NAME,
#                          EXTRACTED_DATA_DIR=EXTRACTED_OUTPUT_PATH, OUTPUT_DIR=VIDEO_PATH)