using Flux
using OceanParameterizations
using WindMixing
using LinearAlgebra
using JLD2

BLAS.set_num_threads(32)

train_files = ["wind_-5e-4_cooling_3e-8_new"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")
# OUTPUT_PATH = joinpath("C:\\Users\\xinle\\Documents\\OceanParameterizations.jl")

FILE_NAME_uw = "uw_NN_training_test"
FILE_NAME_vw = "vw_NN_training_test"
FILE_NAME_wT = "wT_NN_training_test"

FILE_PATH_uw = joinpath(OUTPUT_PATH, "$(FILE_NAME_uw).jld2")
FILE_PATH_vw = joinpath(OUTPUT_PATH, "$(FILE_NAME_vw).jld2")
FILE_PATH_wT = joinpath(OUTPUT_PATH, "$(FILE_NAME_wT).jld2")

EXTRACTED_FILE_PATH = joinpath(PATH, "extracted_training_output")
EXTRACTED_FILE_PATH_uw = joinpath(EXTRACTED_FILE_PATH, "$(FILE_NAME_uw)_extracted.jld2")
EXTRACTED_FILE_PATH_vw = joinpath(EXTRACTED_FILE_PATH, "$(FILE_NAME_vw)_extracted.jld2")
EXTRACTED_FILE_PATH_wT = joinpath(EXTRACTED_FILE_PATH, "$(FILE_NAME_wT)_extracted.jld2")


train_parameters = Dict("ν₀" => 1f-4, "ν₋" => 0.1f0, "Riᶜ" => 0.25f0, "ΔRi" => 1f-1, "Pr" => 1f0, "κ" => 10f0,
                        "modified_pacanowski_philander" => true, "convective_adjustment" => false,
                        "smooth_profile" => false, "smooth_NN" => false, "smooth_Ri" => false, "train_gradient" => true,
                        "zero_weights" => true, "unscaled" => false)

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

N_inputs = 96
hidden_units = 400
N_outputs = 31

uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs))

# weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

# uw_NN = re(weights ./ 1f5)
# vw_NN = re(weights ./ 1f5)
# wT_NN = re(weights ./ 1f5)

# train_optimizers = [ADAM(0.01), Descent()]
# train_epochs = [50,100]

train_optimizers = [ADAM(5e-4), ADAM(2e-4)]
train_epochs = [200, 200]

write_metadata_NN_training(FILE_PATH_uw, train_files, train_parameters, train_epochs, train_optimizers, uw_NN, "uw")
uw_weights = train_NN(uw_NN, 𝒟train, train_optimizers, train_epochs, FILE_PATH_uw, "uw", 
                      modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                      convective_adjustment=train_parameters["convective_adjustment"],
                      ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                      κ=train_parameters["κ"],
                      smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], 
                      train_gradient=train_parameters["train_gradient"],
                      zero_weights = train_parameters["zero_weights"])

extract_NN(FILE_PATH_uw, EXTRACTED_FILE_PATH_uw, "NN")

write_metadata_NN_training(FILE_PATH_vw, train_files, train_parameters, train_epochs, train_optimizers, vw_NN, "vw")
vw_weights = train_NN(vw_NN, 𝒟train.uvT_scaled, 𝒟train.vw.scaled, train_optimizers, train_epochs, FILE_PATH_vw, "vw")

extract_NN(FILE_PATH_vw, EXTRACTED_FILE_PATH_vw, "NN")

write_metadata_NN_training(FILE_PATH_wT, train_files, train_parameters, train_epochs, train_optimizers, wT_NN, "wT")
wT_weights = train_NN(wT_NN, 𝒟train.uvT_scaled, 𝒟train.wT.scaled, train_optimizers, train_epochs, FILE_PATH_wT, "wT")

extract_NN(FILE_PATH_wT, EXTRACTED_FILE_PATH_wT, "NN")
