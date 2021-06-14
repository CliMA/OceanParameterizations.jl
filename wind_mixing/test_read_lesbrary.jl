using JLD2
using FileIO

file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2")
# file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2")

keys(file["parameters/boundary_condition_θ_top"])
file["parameters/boundary_condition_θ_top"](0,0,0)

@inline Qᶿ(x, y, t) = Qᵇ * sin(2π / (Qᵇ_period * 60 ^ 2) * t) / (α * g)
file["parameters/buoyancy_flux"]
file["parameters/momentum_flux"]

file["timeseries/T/383"][1,1,:]
file["timeseries/wu/383"][1,1,:]
close(file)

newfile = jldopen("D:\\Downloads\\test_functions.jld2", "w") do file
    file["function"] = Qᶿ
end

Qᵇ = 3
new_file = jldopen("D:\\Downloads\\test_functions.jld2")
new_file["function"](0,0,1)
