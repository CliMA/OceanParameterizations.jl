using WindMixing
using JLD2
using Plots

train_files = ["diurnal"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
𝒟train.T.coarse
𝒟train.t

file = jldopen("Data/three_layer_constant_fluxes_linear_hr192_Qu5.0e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_diurnal_8days_statistics.jld2")

keys(file["timeseries/wT"])
file["timeseries/wT/1973"][:]
file["timeseries/wT/serialized/location"]
VIDEO_NAME = "u_v_T_3sim_-1e-3_-8e-4_-5e-4_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_2e-4_test_-1e-3"
VIDEO_NAME = "test_video"

plot(file["timeseries/wT/292"][:], -128:1:0, label="")

close(file)

animate_training_data_profiles_fluxes(train_files, joinpath(FILE_PATH, VIDEO_NAME))

function a(x, y)
    x-y
end

file = jldopen("D:\\Downloads\\test.jld2", "w") do file
    file["function"] = a(1,2)
end

read_file = jldopen("D:\\Downloads\\test.jld2")
read_file["function"]

read_file["function"](1,2)

close(read_file)