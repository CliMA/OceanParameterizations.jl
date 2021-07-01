using Images, FileIO
using ImageTransformations
using CairoMakie
using JLD2
using WindMixing: plot_T_profiles_final

u_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\u.png"))))
v_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\v.png"))))
T_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\T.png"))))
uw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\uw.png"))))
vw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\vw.png"))))
wT_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\wT.png"))))
Ri_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\Ri.png"))))
z_img = load(assetpath(joinpath(pwd(), "images_plot\\z.png")))
T_3D_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot\\T_3D.png"))))

axis_images = (
    u = u_img,
    v = v_img,
    T = T_img,
    z = z_img,
    uw = uw_img,
    vw = vw_img,
    wT = wT_img,
    Ri = Ri_img,
    T_3D = T_3D_img,
)

RESULTS_DIR = "C:\\Users\\xinle\\Documents\\OceanParameterizations.jl\\final_results"
NDE_DIR = "18sim_old"

function open_NDE_profile(filepath)
    file = jldopen(filepath)
    NDE_profile = file["NDE_profile"]
    close(file)
    return NDE_profile
end

files_training = [
    "train_wind_-5e-4_cooling_3e-8_new"
    "train_wind_-2e-4_cooling_3e-8_new"
    "train_wind_-5e-4_cooling_1e-8_new"

    "train_wind_-5e-4_heating_-3e-8_new"
    "train_wind_-2e-4_heating_-3e-8_new"
    "train_wind_-5e-4_heating_-1e-8_new"
]

NDE_profiles_training = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_training
]

subplot_titles_training = [
    "Strong Wind, Strong Cooling"
    "Weak Wind, Strong Cooling"
    "Strong Wind, Weak Cooling"

    "Strong Wind, Strong Heating"
    "Weak Wind, Strong Heating"
    "Strong Wind, Weak Heating"
]

frame = 1009

plot_T_profiles_final(NDE_profiles_training, frame, subplot_titles_training, axis_images, "final_results\\constant_training_T.png")

function calculate_losses(NDE_profiles)
    @inline T_loss(data) = sum(data) / 1153

    return [
        (
            NN = T_loss(profile["T_losses"]),
            mpp = T_loss(profile["T_losses_modified_pacanowski_philander"]),
            kpp = T_loss(profile["T_losses_kpp"]),
        ) for profile in NDE_profiles
    ]
end

calculate_losses(NDE_profiles_training)

files_interpolating = [
    "test_wind_-4.5e-4_cooling_2.5e-8"
    "test_wind_-2.5e-4_cooling_2.5e-8"
    "test_wind_-4.5e-4_cooling_1.5e-8"

    "test_wind_-4.5e-4_heating_-2.5e-8"
    "test_wind_-2.5e-4_heating_-2.5e-8"
    "test_wind_-4.5e-4_heating_-1.5e-8"
]

NDE_profiles_interpolating = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_interpolating
]

subplot_titles_interpolating = [
    "Strong Wind, Strong Cooling"
    "Weak Wind, Strong Cooling"
    "Strong Wind, Weak Cooling"

    "Strong Wind, Strong Heating"
    "Weak Wind, Strong Heating"
    "Strong Wind, Weak Heating"
]

frame = 1009

plot_T_profiles_final(NDE_profiles_interpolating, frame, subplot_titles_interpolating, axis_images, "final_results\\constant_interpolating_T.png")

calculate_losses(NDE_profiles_interpolating)


files_extrapolating = [
    "test_wind_-5.5e-4_diurnal_5.5e-8"
    "test_wind_-1.5e-4_cooling_3.5e-8"
    "test_wind_-5.5e-4_new"

    "test_wind_-1.5e-4_diurnal_5.5e-8"
    "test_wind_-1.5e-4_heating_-3.5e-8"
    "test_wind_-5.5e-4_cooling_3.5e-8"
]

NDE_profiles_extrapolating = [
    open_NDE_profile(joinpath(RESULTS_DIR, NDE_DIR, file, "profiles_fluxes_oceananigans.jld2")) for file in files_extrapolating
]

subplot_titles_extrapolating = [
    "Strong Wind, Strong Cycle"
    "Weak Wind, Strong Cooling"
    "Strong Wind Mixing"

    "Weak Wind, Strong Cycle"
    "Weak Wind, Strong Heating"
    "Strong Wind, Strong Cooling"
]

frame = 1009

plot_T_profiles_final(NDE_profiles_extrapolating, frame, subplot_titles_extrapolating, axis_images, "final_results\\constant_extrapolating_T.png")

calculate_losses(NDE_profiles_extrapolating)
