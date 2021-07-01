using Images, FileIO
using ImageTransformations
using CairoMakie
using JLD2
using WindMixing: plot_profiles_fluxes_final_LES

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

filename = "train_wind_-5e-4_heating_-3e-8_new"

file = jldopen(joinpath(RESULTS_DIR, NDE_DIR, filename, "profiles_fluxes_oceananigans.jld2"))

plot_data = file["NDE_profile"]

close(file)

frame = 1009

plot_profiles_fluxes_final_LES(plot_data, frame, axis_images, "text_profiles_fluxes_LES.png")