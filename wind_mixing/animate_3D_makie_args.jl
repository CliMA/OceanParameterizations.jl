using WindMixing: animate_LES_3D
using Images, FileIO
using ImageTransformations
using CairoMakie

simulation = ARGS[1]
num_frames = parse(Int, ARGS[2])
fps = parse(Int, ARGS[3])

u_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "u.png"))))
v_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "v.png"))))
T_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "T.png"))))
uw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "uw.png"))))
vw_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "vw.png"))))
wT_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "wT.png"))))
Ri_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "Ri.png"))))
z_img = load(assetpath(joinpath(pwd(), "images_plot", "z.png")))
T_3D_img = rotr90(load(assetpath(joinpath(pwd(), "images_plot", "T_3D.png"))))

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

DATA_DIR = joinpath(pwd(), "Data")

if simulation == "FC"
    FILE_DIR = joinpath(DATA_DIR, "three_layer_constant_fluxes_linear_hr144_Qu0.0e+00_Qb5.0e-08_f1.0e-04_Nh256_Nz128_free_convection")
    simulation_str = "Free Convection (Surface Cooling)"
elseif simulation == "WM"
    FILE_DIR = joinpath(DATA_DIR, "three_layer_constant_fluxes_linear_hr144_Qu7.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind")
    simulation_str = "Wind-Mixing"
elseif simulation == "WWSC"
    FILE_DIR = joinpath(DATA_DIR, "three_layer_constant_fluxes_linear_hr144_Qu2.2e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling")
    simulation_str = "Weak Wind + Strong Cooling"
else
    FILE_DIR = joinpath(DATA_DIR, "three_layer_constant_fluxes_linear_hr144_Qu5.5e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling")
    simulation_str = "Strong Wind + Weak Cooling"
end

OUTPUT_PATH = joinpath(pwd(), "LES_3D_video", "LES_$(simulation)_$(num_frames)_$(fps)")

if num_frames == 500
    animate_LES_3D(FILE_DIR, OUTPUT_PATH, axis_images, fps=fps, simulation_str=simulation_str, num_frames=num_frames)
else
    animate_LES_3D(FILE_DIR, OUTPUT_PATH, axis_images, fps=fps, simulation_str=simulation_str)
end