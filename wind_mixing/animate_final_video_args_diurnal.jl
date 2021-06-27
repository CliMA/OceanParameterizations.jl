using CairoMakie
using Images, FileIO
using ImageTransformations
using WindMixing
using JLD2
using WindMixing: animate_profiles_fluxes_final

# convective_adjustment = parse(Bool, ARGS[1])
convective_adjustment = false
num = parse(Int, ARGS[1])

if convective_adjustment
    CA_str = "CA"
else
    CA_str = "noCA"
end

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

EXTRACTED_FILE_DIR = joinpath(pwd(), "extracted_training_output")

EXTRACTED_FILE_PATH = joinpath(EXTRACTED_FILE_DIR, "NDE_3sim_diurnal_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8_extracted.jld2")
FILE_DIR = joinpath(pwd(), "final_data_2", "NDE_3sim_diurnal_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8_$(CA_str)")

# if !ispath(FILE_DIR)
#     mkdir(FILE_DIR)
# end

file = jldopen(EXTRACTED_FILE_PATH)
train_files = file["training_info/train_files"]
loss_scalings = file["training_info/loss_scalings"]
train_parameters = file["training_info/parameters"]
close(file)

# test_files_all = [
#     "wind_-5e-4_diurnal_5e-8",    
#     "wind_-3.5e-4_diurnal_5e-8",          
#     "wind_-2e-4_diurnal_5e-8",    

#     "wind_-4.5e-4_diurnal_4e-8",
#     "wind_-4.5e-4_diurnal_2e-8",
#     "wind_-3e-4_diurnal_4e-8",  
#     "wind_-3e-4_diurnal_2e-8",  
# ]

test_files_all = [
    "wind_-5.5e-4_diurnal_5.5e-8" 
    "wind_-1.5e-4_diurnal_5.5e-8" 

    "wind_-5.5e-4_new"            

    "wind_-5.5e-4_heating_-3.5e-8"
    "wind_-1.5e-4_heating_-3.5e-8"
    "wind_-5.5e-4_cooling_3.5e-8" 
    "wind_-1.5e-4_cooling_3.5e-8" 
]

test_files = [test_files_all[num]]

ν₀ = train_parameters["ν₀"]
ν₋ = train_parameters["ν₋"]
ΔRi = train_parameters["ΔRi"]
Riᶜ = train_parameters["Riᶜ"]
Pr = train_parameters["Pr"]

solve_oceananigans_modified_pacanowski_philander_nn(test_files, EXTRACTED_FILE_PATH, FILE_DIR,
                                                        timestep=1, convective_adjustment=convective_adjustment)

diurnal = occursin("diurnal", test_files[1])

if test_files[1] in train_files
    DIR_NAME = "train_$(test_files[1])"
    animation_type = "Training"
else
    DIR_NAME = "test_$(test_files[1])"
    # animation_type = "Interpolating"
    animation_type = "Extrapolating"
end

n_trainings = length(train_files)

plot_data = NDE_profile_oceananigans(joinpath(FILE_DIR, DIR_NAME), train_files, test_files,
                                  ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr, 
                                  loss_scalings=loss_scalings,
                                  OUTPUT_PATH=joinpath(FILE_DIR, DIR_NAME, "profiles_fluxes_oceananigans.jld2"))

animate_profiles_fluxes_final(plot_data, axis_images, 
                    joinpath(FILE_DIR, DIR_NAME, "$(test_files[1])"),
                    animation_type=animation_type, n_trainings=n_trainings, training_types="Wind + Diurnal Heating & Cooling")