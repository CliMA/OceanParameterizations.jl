using Statistics
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using WindMixing
using JLD2
using FileIO

PATH = joinpath(pwd(), "extracted_training_output")
# PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"

DATA_NAME = "NDE_training_mpp_8sim_wind_mixing_cooling_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4"
DATA_PATH = joinpath(PATH, "$(DATA_NAME)_extracted.jld2")
ispath(DATA_PATH)
FILE_PATH = joinpath(pwd(), "Output")

file = jldopen(DATA_PATH, "r")

train_files = file["training_info/train_files"]
train_parameters = file["training_info/parameters"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
close(file)

N_inputs = 96
hidden_units = 400
N_outputs = 31

weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

uw_NN = re(zeros(Float32, length(weights)))
vw_NN = re(zeros(Float32, length(weights)))
wT_NN = re(zeros(Float32, length(weights)))


for test_file in keys(directories)
    @info "running $test_file"
    test_files = [test_file]
    𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
    trange = 1:1:1153
    plot_data = NDE_profile_mutating(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange,
                            modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                            ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], 
                            Riᶜ=train_parameters["Riᶜ"], convective_adjustment=train_parameters["convective_adjustment"],
                            smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"],
                            zero_weights=train_parameters["zero_weights"],
                            gradient_scaling=train_parameters["gradient_scaling"])

    animation_type = "Pre-Training"
    n_trainings = length(train_files)
    training_types = "Modified Pacanowski-Philander"
    VIDEO_NAME = "$(test_file)_mpp"
    animate_profiles_fluxes_comparison(plot_data, joinpath(FILE_PATH, VIDEO_NAME), fps=30, 
                                                    animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)
end