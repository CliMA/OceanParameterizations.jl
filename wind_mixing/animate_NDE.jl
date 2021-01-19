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

DATA_PATH = joinpath(PATH, "extracted_training_output", "NDE_training_strong_convective_adjustment_1sim_-1e-3_extracted.jld2")

file = jldopen(DATA_PATH, "r")

losses = file["losses"]

minimum(losses)
size = length(losses)

train_files = file["training_info/train_files"]

plot(1:1:size, losses, yscale=:log10)
xlabel!("Iteration")
ylabel!("Loss mse")
savefig(joinpath(PATH, "Output", "NDE_training_strong_convective_adjustment_1sim_-1e-3_loss.pdf"))
𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

test_files = ["-1e-3"]
𝒟test = data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

trange = 1:1:1153
plot_data = NDE_profile_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange)

FILE_PATH = joinpath(pwd(), "Output")

animate_profile_flux(plot_data, "u", "uw", joinpath(FILE_PATH, "u_uw_strong_convective_adjustment_1sim_-1e-3_2_temp"))
animate_profile_flux(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_strong_convective_adjustment_1sim_-1e-3_2_temp"))
animate_profile_flux(plot_data, "T", "wT", joinpath(FILE_PATH, "w_wT_strong_convective_adjustment_1sim_-1e-3_2_temp"))

# animate_profile(plot_data, "u", joinpath(FILE_PATH, "u_test"))
# animate_profile(plot_data, "v", joinpath(FILE_PATH, "v_test"))
# animate_profile(plot_data, "T", joinpath(FILE_PATH, "T_test"))

# animate_flux(plot_data, "uw", joinpath(FILE_PATH, "uw_test"))
# animate_flux(plot_data, "vw", joinpath(FILE_PATH, "vw_test"))
# animate_flux(plot_data, "wT", joinpath(FILE_PATH, "wT_test"))
