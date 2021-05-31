using JLD2
using FileIO

file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu0.0e+00_Qb4.0e-08_f1.0e-04_Nh256_Nz128_free_convection_8days_statistics.jld2")
# file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_wind_mixing_8days_2_statistics.jld2")
file = jldopen("Data/instantaneous_statistics_with_halos_view.jld2")


file["parameters/buoyancy_flux"]
file["parameters/momentum_flux"]

file["timeseries/T/383"][1,1,:]
file["timeseries/wu/383"][1,1,:]
close(file)