using FileIO: Base
using JLD2
using Plots
using WindMixing
using OceanParameterizations

PATH = pwd()
FILE_PATH_NN = joinpath(PATH, "extracted_training_output", 
        "NDE_training_modified_pacanowski_philander_1sim_-1e-3_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4_extracted.jld2")

ds = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2")

@assert isfile(FILE_PATH_NN)
file = jldopen(FILE_PATH_NN, "r")

uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]

f = ds["parameters/coriolis_parameter"]
α = ds["parameters/thermal_expansion_coefficient"]
g = ds["parameters/gravitational_acceleration"]
Nz = 32
Lz = ds["grid/Lz"]
Δz = ds["grid/Δz"]

uw_flux = ds["parameters/boundary_condition_u_top"]
vw_flux = 0
wT_flux = ds["parameters/boundary_condition_θ_top"]

T₀ = Array(ds["timeseries/T/0"][1, 1, :])

∂u₀∂z = ds["parameters/boundary_condition_u_bottom"]
∂v₀∂z = ds["parameters/boundary_condition_u_bottom"]
∂T₀∂z = ds["parameters/boundary_condition_θ_bottom"]

train_files = file["training_info/train_files"]

𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

u_scaling = 𝒟.scalings["u"]
v_scaling = 𝒟.scalings["v"]
T_scaling = 𝒟.scalings["T"]
uw_scaling = 𝒟.scalings["uw"]
vw_scaling = 𝒟.scalings["vw"]
wT_scaling = 𝒟.scalings["wT"]

scalings = (u=u_scaling, v=v_scaling, T=T_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling)

constants = (; f, α, g, Nz, Lz, T₀)
BCs = (top=(uw=uw_flux, vw=vw_flux, wT=wT_flux), bottom=(u=∂u₀∂z, v=∂v₀∂z, T=∂T₀∂z))

oceananigans_modified_pacanowski_philander_nn(constants, BCs, scalings, NN_filepath=FILE_PATH_NN, stop_time=691200, Δt=120)
close(file)
close(ds)


baseline_result = jldopen(joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl", "oceananigans_baseline.jld2"))
NN_result = jldopen(joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl", "oceananigans_modified_pacanowski_philander_NN.jld2"))

Nz = baseline_result["grid"]
frames = keys(baseline_result["timeseries/t"])
t = [baseline_result["timeseries/t/$(frames[i])"] for i in 1:length(frames)]

plot(baseline_result["timeseries/T/60"][1,1,:], baseline_result["grid/zC"][2:end-1], label = "")

bottom_T = [baseline_result["timeseries/T/$i"][1,1,1] for i in 0:60]

bottom_T = [ds["timeseries/T/$(keys(ds["timeseries/T"])[i])"][1,1,1] for i in 1:60]

plot(1:60, bottom_T)
xlabel!("time")
ylabel!("Temperature")
close(baseline_result)
close(NN_result)
