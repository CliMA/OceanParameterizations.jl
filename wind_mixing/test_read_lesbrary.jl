using JLD2
using FileIO
using WindMixing
using OceanParameterizations
using Plots


train_files = ["wind_-5e-4_diurnal_3e-8"]

𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)

𝒟.wT.coarse

plot(1:length(𝒟.wT.coarse[end,:]), 𝒟.wT.coarse[end,:])

file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb3.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2")
# file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2")

keys(file["parameters/boundary_condition_θ_top"])
file["parameters/boundary_condition_θ_top"]
wb = file["parameters/buoyancy_flux"]
α = file["parameters/thermal_expansion_coefficient"]
g = file["parameters/gravitational_acceleration"]

wb / α / g ≈ file["parameters/boundary_condition_θ_top"]

@inline Qᶿ(x, y, t) = Qᵇ * sin(2π / (Qᵇ_period * 60 ^ 2) * t) / (α * g)
file["parameters/buoyancy_flux"]
file["parameters/momentum_flux"]

keys(file["timeseries/T"])

file["timeseries/T/$(keys(file["timeseries/T"])[5])"][1,1,:]
file["timeseries/wT/$(keys(file["timeseries/T"])[2])"][1,1,end]
close(file)

newfile = jldopen("D:\\Downloads\\test_functions.jld2", "w") do file
    file["function"] = Qᶿ
end

Qᵇ = 3
new_file = jldopen("D:\\Downloads\\test_functions.jld2")
new_file["function"](0,0,1)

train_files = ["diurnal_test"]
𝒟 = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)