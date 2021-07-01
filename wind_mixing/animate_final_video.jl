using CairoMakie
using Images, FileIO
using ImageTransformations
using WindMixing
using JLD2
using WindMixing: animate_profiles_fluxes_final

u_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "u.png"))))
v_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "v.png"))))
T_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "T.png"))))
uw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "uw.png"))))
vw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "vw.png"))))
wT_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "wT.png"))))
Ri_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "Ri.png"))))
z_img = load(assetpath(joinpath(pwd(), "images_plot", "z.png")))

axis_images = (
    u = u_img,
    v = v_img,
    T = T_img,
    z = z_img,
    uw = uw_img,
    vw = vw_img,
    wT = wT_img,
    Ri = Ri_img,
)

# EXTRACTED_FILE_PATH = joinpath(pwd(), 
                        # "extracted_training_output/NDE_18sim_windcooling_windheating_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8_1e-4_extracted.jld2")


EXTRACTED_FILE_PATH = joinpath(pwd(), 
                        "extracted_training_output/NDE_3sim_diurnal_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8_extracted.jld2")

file = jldopen(EXTRACTED_FILE_PATH)
train_files = file["training_info/train_files"]
loss_scalings = file["training_info/loss_scalings"]
train_parameters = file["training_info/parameters"]
close(file)

test_files = [
    # "wind_-5e-4_cooling_3e-8_new",
    # "wind_-2e-4_cooling_3e-8_new",
    # "wind_-2e-4_heating_-3e-8_new",
    # "wind_-5e-4_diurnal_5e-8",
    # "wind_-5e-4_diurnal_1e-8"
    # "cooling_3.5e-8_new",
    # "wind_-5e-4_new"
]

ν₀ = train_parameters["ν₀"]
ν₋ = train_parameters["ν₋"]
ΔRi = train_parameters["ΔRi"]
Riᶜ = train_parameters["Riᶜ"]
Pr = train_parameters["Pr"]

FILE_DIR = joinpath("C:\\Users\\xinle\\Documents\\OceanParameterizations.jl\\test_diurnal")
solve_oceananigans_modified_pacanowski_philander_nn(test_files, EXTRACTED_FILE_PATH, FILE_DIR,
                                                        timestep=200, convective_adjustment=false)

plot_data = NDE_profile_oceananigans(joinpath(FILE_DIR, "test_$(test_files[1])"), train_files, test_files,
                                  ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr, 
                                  loss_scalings=loss_scalings,
                                  OUTPUT_PATH="")

# animate_profiles_fluxes_final(plot_data, axis_images, 
#                     joinpath(FILE_DIR, "test_color"),
#                     animation_type="Training", n_trainings=18, training_types="Wind + Cooling, Wind + Heating", fps=60)