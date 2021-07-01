using WindMixing: animate_LES_3D
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

FILE_DIR = "C:\\Users\\xinle\\Downloads\\three_layer_constant_fluxes_linear_hr144_Qu5.5e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling"
OUTPUT_PATH = "test_LES"
animate_LES_3D(FILE_DIR, OUTPUT_PATH, axis_images, fps=5)