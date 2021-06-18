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
                    # FILE_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(pwd(), "Output")
# VIDEO_NAME = "u_v_T_mpp_2sim_wind_mixing_-1e-3_cooling_5e-8_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4_test_5e-8"
VIDEO_NAME = "test"
# SIMULATION_NAME = "NN Smoothing Wind-Mixing, Testing Data"
SIMULATION_NAME = "4 Simulation Training (Wind Mixing, Free Convection)"

# file = jldopen(DATA_PATH, "r")
file = jldopen(DATA_PATH, "r")

losses = file["losses"]

minimum(losses)

train_files = file["training_info/train_files"]
train_parameters = file["training_info/parameters"]

if haskey(file["training_info"], "loss_scalings")
    loss_scalings = file["training_info/loss_scalings"]
elseif haskey(train_parameters, "gradient_scaling")
    gradient_scaling = train_parameters["gradient_scaling"]
    loss_scalings = (u=1f0, v=1f0, T=1f0, ∂u∂z=gradient_scaling, ∂v∂z=gradient_scaling, ∂T∂z=gradient_scaling)
end


# Plots.plot(1:1:length(losses), losses, yscale=:log10)
# Plots.xlabel!("Iteration")
# Plots.ylabel!("Loss mse")
# savefig(joinpath(PATH, "Output", "NDE_training_modified_pacanowski_philander_1sim_-1e-3_smaller_learning_rate_loss.pdf"))
# train_files = ["-1e-3"]
𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

test_files = ["wind_-1e-3_heating_-4e-8"]
𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]


close(file)
[uw_NN(rand(96)) uw_NN(rand(96))]

# N_inputs = 96
# hidden_units = 400
# N_outputs = 31

# weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

# uw_NN = re(zeros(Float32, length(weights)))
# vw_NN = re(zeros(Float32, length(weights)))
# wT_NN = re(zeros(Float32, length(weights)))


𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
trange = 1:1:1153
plot_data = NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange,
                        modified_pacanowski_philander=train_parameters["modified_pacanowski_philander"], 
                        # ν₀=1f-4, ν₋=0.1f0, ΔRi=1f-1,
                        ν₀=train_parameters["ν₀"], ν₋=train_parameters["ν₋"], ΔRi=train_parameters["ΔRi"], 
                        Riᶜ=train_parameters["Riᶜ"], convective_adjustment=train_parameters["convective_adjustment"],
                        smooth_NN=train_parameters["smooth_NN"], smooth_Ri=train_parameters["smooth_Ri"],
                        zero_weights=train_parameters["zero_weights"],
                        loss_scalings=loss_scalings)

# WindMixing.animate_profiles_fluxes(plot_data, joinpath(FILE_PATH, VIDEO_NAME), dimensionless=false, SIMULATION_NAME=SIMULATION_NAME)

animation_type = "Training"
n_trainings = length(train_files)
training_types = "Wind Mixing, Free Convection"
VIDEO_NAME = "test"
animate_profiles_fluxes_comparison(plot_data, plot_data, plot_data, joinpath(FILE_PATH, VIDEO_NAME), fps=30, 
                                                animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)


# VIDEO_NAME = "u_v_T_modified_pacanowski_philander_1sim_-1e-3_test2"

# keys(plot_data)

# plot_data["truth_T"][:,1]

# uvT_truth = [plot_data["truth_u"]; plot_data["truth_v"]; plot_data["truth_T"]]
# Ris = local_richardson(uvT_truth, 𝒟test, unscale=true)

# animate_local_richardson_profile(uvT_truth, 𝒟test, joinpath(FILE_PATH, "Ris_convective_adjustment_1sim_-1e-3_2_test"), unscale=true)

# plot(Ris[:,3], plot_data["depth_flux"])
# xlabel!("Ri")
# ylabel!("z")

# animate_profile_flux(plot_data, "u", "uw", joinpath(FILE_PATH, "u_uw_modified_pacanowski_philander_1sim_-1e-3_test"), gif=true, dimensionless=false)
# animate_profile_flux(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_modified_pacanowski_philander_1sim_-1e-3_test"), gif=true, dimensionless=false)
# animate_profile_flux(plot_data, "T", "wT", joinpath(FILE_PATH, "w_wT_modified_pacanowski_philander_1sim_-1e-3_test"), gif=true, dimensionless=false)

# animate_profiles(plot_data, joinpath(FILE_PATH, VIDEO_NAME), dimensionless=false)

# animate_profile(plot_data, "u", "uw", joinpath(FILE_PATH, "u_uw_convective_adjustment_viscosity_empty"), gif=true)
# animate_profile(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_convective_adjustment_viscosity_empty"), gif=true)
# animate_profile(plot_data, "T", "wT", joinpath(FILE_PATH, "w_wT_convective_adjustment_viscosity_empty"), gif=true)

# animate_flux(plot_data, "uw", joinpath(FILE_PATH, "uw_test"))
# animate_flux(plot_data, "vw", joinpath(FILE_PATH, "vw_test"))
# animate_flux(plot_data, "wT", joinpath(FILE_PATH, "wT_test"))

# train_files = ["-1e-3", "-9e-4", "-8e-4", "-7e-4", "-6e-4", "-5e-4", "-4e-4", "-3e-4", "-2e-4"]
train_files = ["wind_-1e-3_heating_-4e-8"]
# train_files = ["cooling_1e-8"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
𝒟train.T.coarse
𝒟train.t
VIDEO_NAME = "u_v_T_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4_test_-1e-3"
VIDEO_NAME = "test_video"

animate_training_data_profiles_fluxes(train_files, joinpath(FILE_PATH, VIDEO_NAME))

test_files = ["-1e-3", "cooling_5e-8", "-8e-4", "cooling_4e-8"]

animate_training_results(test_files, DATA_NAME, trange=1:1:1153)