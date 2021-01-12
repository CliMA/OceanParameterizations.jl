using Statistics
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
using WindMixing

PATH = pwd()

# data used to train the neural differential equations
train_files = ["strong_wind"]
uw_NN = BSON.load(joinpath(PATH, "NDEs", "uw_NDE_convective_adjustment_100_large.bson"))[:neural_network]
vw_NN = BSON.load(joinpath(PATH, "NDEs", "vw_NDE_convective_adjustment_100_large.bson"))[:neural_network]
wT_NN = BSON.load(joinpath(PATH, "NDEs", "wT_NDE_convective_adjustment_100_large.bson"))[:neural_network]

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
trange = 1:1:100

plot_data = NDE_profile_convective_adjustment(uw_NN, vw_NN, wT_NN, 𝒟train, 𝒟train, trange)

FILE_PATH = joinpath(pwd(), "Output")

animate_profile_flux(plot_data, "u", "wT", joinpath(FILE_PATH, "u_uw_test"))
animate_profile_flux(plot_data, "v", "vw", joinpath(FILE_PATH, "v_vw_test"))
animate_profile_flux(plot_data, "T", "wT", joinpath(FILE_PATH, "T_wT_test"))

animate_profile(plot_data, "u", joinpath(FILE_PATH, "u_test"))
animate_profile(plot_data, "v", joinpath(FILE_PATH, "v_test"))
animate_profile(plot_data, "T", joinpath(FILE_PATH, "T_test"))

animate_flux(plot_data, "uw", joinpath(FILE_PATH, "uw_test"))
animate_flux(plot_data, "vw", joinpath(FILE_PATH, "vw_test"))
animate_flux(plot_data, "wT", joinpath(FILE_PATH, "wT_test"))
