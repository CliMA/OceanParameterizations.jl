using Flux
using OceanParameterizations
using WindMixing
using LinearAlgebra
using JLD2

# T_fraction_str = ARGS[1]
# T_fraction = parse(Float32, T_fraction_str)

NN_type = ARGS[1]
# N_sims = parse(Int, ARGS[2])
N_sims = 9
# rate_str = ARGS[4]
# rate = parse(Float64, rate_str)
# params_type = ARGS[4]
params_type = "old"

# T_fraction = parse(Float32, "0.8")
# NN_type = "relu"

train_files_all = [
  "wind_-5e-4_diurnal_5e-8",  
  "wind_-5e-4_diurnal_3e-8",  
  "wind_-5e-4_diurnal_1e-8",  
      
  "wind_-3.5e-4_diurnal_5e-8",
  "wind_-3.5e-4_diurnal_3e-8",
  "wind_-3.5e-4_diurnal_1e-8",
      
  "wind_-2e-4_diurnal_5e-8",  
  "wind_-2e-4_diurnal_3e-8",  
  "wind_-2e-4_diurnal_1e-8",  
]

# train_files_all = [
#   "wind_-5e-4_cooling_3e-8_new",   
#   "wind_-5e-4_cooling_1e-8_new",   
#   "wind_-2e-4_cooling_3e-8_new",   
#   "wind_-2e-4_cooling_1e-8_new",   
#   "wind_-5e-4_heating_-3e-8_new",  
#   "wind_-2e-4_heating_-1e-8_new",  
#   "wind_-2e-4_heating_-3e-8_new",  
#   "wind_-5e-4_heating_-1e-8_new",  

#   "wind_-3.5e-4_cooling_2e-8_new", 
#   "wind_-3.5e-4_heating_-2e-8_new",

#   "wind_-5e-4_cooling_2e-8_new",   
#   "wind_-3.5e-4_cooling_3e-8_new", 
#   "wind_-3.5e-4_cooling_1e-8_new", 
#   "wind_-2e-4_cooling_2e-8_new",   
#   "wind_-3.5e-4_heating_-3e-8_new",
#   "wind_-3.5e-4_heating_-1e-8_new",
#   "wind_-2e-4_heating_-2e-8_new",  
#   "wind_-5e-4_heating_-2e-8_new",  
# ]

train_files = train_files_all[1:N_sims]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")
# OUTPUT_PATH = joinpath("C:\\Users\\xinle\\Documents\\OceanParameterizations.jl")

FILE_NAME_uw = "uw_NN_$(N_sims)sim_diurnal_$(params_type)_$(NN_type)_rate_5e-4_2e-4"
FILE_NAME_vw = "vw_NN_$(N_sims)sim_diurnal_$(params_type)_$(NN_type)_rate_5e-4_2e-4"
FILE_NAME_wT = "wT_NN_$(N_sims)sim_diurnal_$(params_type)_$(NN_type)_rate_5e-4_2e-4"

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
                        "zero_weights" => false, "unscaled" => false, "gradient_scaling" => 1f-4)

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

N_inputs = 96
hidden_units = 400
N_outputs = 31

if NN_type == "mish"
  activation = mish
elseif NN_type == "swish"
  activation = swish
elseif NN_type == "leakyrelu"
  activation = leakyrelu
elseif NN_type == "relu"
  activation = relu
else
  activation = tanh
end

uw_NN = Chain(Dense(N_inputs, hidden_units, activation), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, activation), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, activation), Dense(hidden_units, N_outputs))

# weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

# uw_NN = re(weights ./ 1f5)
# vw_NN = re(weights ./ 1f5)
# wT_NN = re(weights ./ 1f5)

# train_optimizers = [ADAM(0.01), Descent()]
# train_epochs = [50,100]

train_optimizers = [ADAM(5e-4), ADAM(2e-4), Descent(2e-4)]
train_epochs = [200, 200, 100]

write_metadata_NN_training(FILE_PATH_uw, train_files, train_parameters, train_epochs, train_optimizers, uw_NN, "uw")
uw_weights = train_NN(uw_NN, 𝒟train, train_optimizers, train_epochs, FILE_PATH_uw, "uw", 
                      modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                      convective_adjustment=train_parameters["convective_adjustment"],
                      ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                      κ=train_parameters["κ"],
                      smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], 
                      train_gradient=train_parameters["train_gradient"],
                      zero_weights = train_parameters["zero_weights"],
                      gradient_scaling=train_parameters["gradient_scaling"])

extract_NN(FILE_PATH_uw, EXTRACTED_FILE_PATH_uw, "NN")

write_metadata_NN_training(FILE_PATH_vw, train_files, train_parameters, train_epochs, train_optimizers, vw_NN, "vw")
vw_weights = train_NN(vw_NN, 𝒟train, train_optimizers, train_epochs, FILE_PATH_vw, "vw", 
                      modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                      convective_adjustment=train_parameters["convective_adjustment"],
                      ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                      κ=train_parameters["κ"],
                      smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], 
                      train_gradient=train_parameters["train_gradient"],
                      zero_weights = train_parameters["zero_weights"],
                      gradient_scaling=train_parameters["gradient_scaling"])

extract_NN(FILE_PATH_vw, EXTRACTED_FILE_PATH_vw, "NN")

write_metadata_NN_training(FILE_PATH_wT, train_files, train_parameters, train_epochs, train_optimizers, wT_NN, "wT")
wT_weights = train_NN(wT_NN, 𝒟train, train_optimizers, train_epochs, FILE_PATH_wT, "wT", 
                      modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                      convective_adjustment=train_parameters["convective_adjustment"],
                      ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                      κ=train_parameters["κ"],
                      smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], 
                      train_gradient=train_parameters["train_gradient"],
                      zero_weights = train_parameters["zero_weights"],
                      gradient_scaling=train_parameters["gradient_scaling"])

extract_NN(FILE_PATH_wT, EXTRACTED_FILE_PATH_wT, "NN")
