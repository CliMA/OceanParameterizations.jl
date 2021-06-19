using Flux
using OceanParameterizations
using WindMixing
using LinearAlgebra

BLAS.set_num_threads(32)

train_files = ["-1e-3"]

PATH = pwd()
OUTPUT_PATH = joinpath(PATH, "training_output")
OUTPUT_PATH = joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output")

# FILE_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "testNN.jld2")

FILE_PATH_uw = joinpath(OUTPUT_PATH, "uw_NN_training_test.jld2")
FILE_PATH_vw = joinpath(OUTPUT_PATH, "vw_NN_training_test.jld2")
FILE_PATH_wT = joinpath(OUTPUT_PATH, "wT_NN_training_test.jld2")

train_parameters = Dict("ν₀" => 1f-4, "ν₋" => 0.1f0, "Riᶜ" => 0.25f0, "ΔRi" => 1f-1, "Pr" => 1f0, "κ" => 10f0,
                        "modified_pacanowski_philander" => true, "convective_adjustment" => false,
                        "smooth_profile" => false, "smooth_NN" => false, "smooth_Ri" => false, "train_gradient" => true,
                        "zero_weights" => true, "unscaled" => false)

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

N_inputs = 96
hidden_units = 50
N_outputs = 31

uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs))

# train_optimizers = [ADAM(0.01), Descent()]
# train_epochs = [50,100]

train_optimizers = [ADAM(5e-3)]
train_epochs = [30]

write_metadata_NN_training(FILE_PATH_uw, train_files, train_epochs, train_optimizers, uw_NN, "uw")
uw_weights = train_NN(uw_NN, 𝒟train, train_optimizers, train_epochs, FILE_PATH_uw, "uw", 
                      modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                      convective_adjustment=train_parameters["convective_adjustment"],
                      ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], Riᶜ=train_parameters["Riᶜ"], 
                      κ=train_parameters["κ"],
                      smooth_profile=train_parameters["smooth_profile"], smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"], 
                      train_gradient=train_parameters["train_gradient"],
                      zero_weights = train_parameters["zero_weights"])

weights, re = Flux.destructure(uw_NN)

uw_weights
uw_NN = re(uw_weights)
[uw_NN(𝒟train.uvT_scaled[:,1]) uw_NN(𝒟train.uvT_scaled[:,100]) uw_NN(𝒟train.uvT_scaled[:,500]) uw_NN(rand(96))]

uw_NN(𝒟train.uvT_scaled[:,1]) .- uw_NN(𝒟train.uvT_scaled[:,1000])

write_metadata_NN_training(FILE_PATH_vw, train_files, train_epochs, train_optimizers, vw_NN, "vw")
uw_weights = train_NN(vw_NN, 𝒟train.uvT_scaled, 𝒟train.vw.scaled, train_optimizers, train_epochs, FILE_PATH_vw, "vw")

write_metadata_NN_training(FILE_PATH_wT, train_files, train_epochs, train_optimizers, wT_NN, "wT")
uw_weights = train_NN(wT_NN, 𝒟train.uvT_scaled, 𝒟train.wT.scaled, train_optimizers, train_epochs, FILE_PATH_wT, "wT")



# file = jldopen(FILE_PATH, "r")
# close(file)