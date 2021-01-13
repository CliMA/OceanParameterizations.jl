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

DATA_PATH = joinpath(PATH, "training_output", "NDE_training_1sim_convective_adjustment_temp2.jld2")

file = jldopen(DATA_PATH, "r")

# data used to train the neural differential equations
# uw_NN = BSON.load(joinpath(PATH, "NDEs", "uw_NDE_convective_adjustment_100_large.bson"))[:neural_network]
# vw_NN = BSON.load(joinpath(PATH, "NDEs", "vw_NDE_convective_adjustment_100_large.bson"))[:neural_network]
# wT_NN = BSON.load(joinpath(PATH, "NDEs", "wT_NDE_convective_adjustment_100_large.bson"))[:neural_network]

train_files = ["-1e-3"]
file["loss/5000"]
size = length(keys(file["loss"]))
index = parse.(Int, keys(file["loss"]))
start_index = minimum(index) - 1
losses = Array{Float32}(undef, size)

for i in 1:size
    losses[i] = file["loss/$(i + start_index)"]
end

minimum(losses)

plot(1:1:size, losses)
𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
trange = 1:1:500

test_files = ["-1e-3"]
𝒟test = data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
uw_NN = file["neural_network/uw/$(argmin(losses) + start_index)"]
vw_NN = file["neural_network/vw/$(argmin(losses) + start_index)"]
wT_NN = file["neural_network/wT/$(argmin(losses) + start_index)"]


NN_PATH = joinpath("D:\\University Matters\\Massachusetts Institute of Technology\\CLiMA Project\\OceanParameterizations.jl\\training_output", "NDE_convective_adjustment_1sim_-1e-3_t200.jld2")

write_data_NN(NN_PATH, uw_NN, vw_NN, wT_NN)

plot_data = NDE_profile_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟test, 𝒟train, trange)

FILE_PATH = joinpath(pwd(), "Output")

animate_profile_flux(plot_data, "u", "uw", joinpath(FILE_PATH, "u_uw_-1e-3_temp"))
animate_profile_flux(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_-1e-3_temp"))
animate_profile_flux(plot_data, "T", "wT", joinpath(FILE_PATH, "w_wT_-1e-3_temp"))

# animate_profile(plot_data, "u", joinpath(FILE_PATH, "u_test"))
# animate_profile(plot_data, "v", joinpath(FILE_PATH, "v_test"))
# animate_profile(plot_data, "T", joinpath(FILE_PATH, "T_test"))

# animate_flux(plot_data, "uw", joinpath(FILE_PATH, "uw_test"))
# animate_flux(plot_data, "vw", joinpath(FILE_PATH, "vw_test"))
# animate_flux(plot_data, "wT", joinpath(FILE_PATH, "wT_test"))
