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

PATH = pwd()

# DATA_PATH = joinpath(PATH, "extracted_training_output", "NDE_training_modified_pacalowski_philander_1sim_-1e-3_2_extracted.jld2")
DATA_PATH = joinpath(PATH, "extracted_training_output", "NDE_training_modified_pacalowski_philander_1sim_-1e-3_smaller_learning_rate_extracted.jld2")
# FILE_PATH = "D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(PATH, "Output")
VIDEO_NAME = "u_v_T_modified_pacalowski_philander_1sim_-1e-3_smaller_learning_rate_test_-9e-4"

file = jldopen(DATA_PATH, "r")

losses = file["losses"]

minimum(losses)
size = length(losses)

train_files = file["training_info/train_files"]

plot(1:1:size, losses, yscale=:log10)
xlabel!("Iteration")
ylabel!("Loss mse")
savefig(joinpath(PATH, "Output", "NDE_training_modified_pacalowski_philander_1sim_-1e-3_smaller_learning_rate_loss.pdf"))

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

test_files = ["-9e-4"]
𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

# uw_weights, re_uw = Flux.destructure(uw_NN)
# vw_weights, re_vw = Flux.destructure(vw_NN)
# wT_weights, re_wT = Flux.destructure(wT_NN)

# uw_weights .= 0f0

# uw_NN = re_uw(uw_weights)
# vw_NN = re_vw(uw_weights)
# wT_NN = re_wT(uw_weights)

trange = 1:1:1153
plot_data = NDE_profile(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange, unscale=true, modified_pacalowski_philander=true, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0)

animate_profiles_fluxes(plot_data, joinpath(FILE_PATH, VIDEO_NAME), dimensionless=false, SIMULATION_NAME="Modified Pacalowski-Philander Wind-Mixing, Training Data")


# VIDEO_NAME = "u_v_T_modified_pacalowski_philander_1sim_-1e-3_test2"


# keys(plot_data)

# plot_data["truth_T"][:,1]

# uvT_truth = [plot_data["truth_u"]; plot_data["truth_v"]; plot_data["truth_T"]]
# Ris = local_richardson(uvT_truth, 𝒟test, unscale=true)

# animate_local_richardson_profile(uvT_truth, 𝒟test, joinpath(FILE_PATH, "Ris_convective_adjustment_1sim_-1e-3_2_test"), unscale=true)

# plot(Ris[:,3], plot_data["depth_flux"])
# xlabel!("Ri")
# ylabel!("z")


# animate_profile_flux(plot_data, "u", "uw", joinpath(FILE_PATH, "u_uw_modified_pacalowski_philander_1sim_-1e-3_test"), gif=true, dimensionless=false)
# animate_profile_flux(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_modified_pacalowski_philander_1sim_-1e-3_test"), gif=true, dimensionless=false)
# animate_profile_flux(plot_data, "T", "wT", joinpath(FILE_PATH, "w_wT_modified_pacalowski_philander_1sim_-1e-3_test"), gif=true, dimensionless=false)

# animate_profiles(plot_data, joinpath(FILE_PATH, VIDEO_NAME), dimensionless=false)

# animate_profile(plot_data, "u", "uw", joinpath(FILE_PATH, "u_uw_convective_adjustment_viscosity_empty"), gif=true)
# animate_profile(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_convective_adjustment_viscosity_empty"), gif=true)
# animate_profile(plot_data, "T", "wT", joinpath(FILE_PATH, "w_wT_convective_adjustment_viscosity_empty"), gif=true)

# animate_flux(plot_data, "uw", joinpath(FILE_PATH, "uw_test"))
# animate_flux(plot_data, "vw", joinpath(FILE_PATH, "vw_test"))
# animate_flux(plot_data, "wT", joinpath(FILE_PATH, "wT_test"))
