using WindMixing: plot_LES_3D
using Images, FileIO
using ImageTransformations
using CairoMakie

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

# FILE_DIR = "C:\\Users\\xinle\\Downloads\\three_layer_constant_fluxes_linear_hr144_Qu5.5e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling"
# FILE_DIR = "C:\\Users\\xinle\\Downloads\\three_layer_constant_fluxes_linear_hr144_Qu0.0e+00_Qb5.0e-08_f1.0e-04_Nh256_Nz128_free_convection"
# FILE_DIR = "C:\\Users\\xinle\\Downloads\\three_layer_constant_fluxes_linear_hr144_Qu7.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind"
FILE_DIR = "Data\\three_layer_constant_fluxes_linear_hr144_Qu2.2e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling"

frame = 145
OUTPUT_PATH = "final_results\\test_LES.png"

# file = jldopen(joinpath(FILE_DIR, "xz_slice.jld2"))

# close(file)
plot_LES_3D(frame, FILE_DIR, OUTPUT_PATH, axis_images, title="", colorscheme=:turbo, rev=false)