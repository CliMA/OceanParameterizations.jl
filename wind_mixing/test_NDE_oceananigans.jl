using FileIO: Base
using JLD2
using Plots
using WindMixing
using OceanParameterizations

PATH = pwd()

EXTRACTED_FILE = "NDE_training_modified_pacanowski_philander_1sim_-1e-3_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4_extracted.jld2"
EXTRACTED_FILE_PATH = joinpath(PATH, "extracted_training_output", EXTRACTED_FILE)

OUTPUT_DIR = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl"
test_files = ["-1e-3"]
solve_oceananigans_modified_pacanowski_philander_nn(test_files, EXTRACTED_FILE_PATH, OUTPUT_DIR, timestep=60)

file = jldopen(EXTRACTED_FILE_PATH)
train_files = file["training_info/train_files"]
gradient_scaling = file["training_info/parameters"]["gradient_scaling"]

OUTPUT_SOL_DIR = joinpath(OUTPUT_DIR, "train_$(test_files[1])")
plot_data = NDE_profile_oceananigans(OUTPUT_SOL_DIR, train_files, test_files,
                                  ν₀=1f-1, ν₋=1f-4, ΔRi=1f-1, Riᶜ=0.25f0, Pr=1, gradient_scaling=gradient_scaling,
                                  OUTPUT_PATH=joinpath(OUTPUT_SOL_DIR, "profiles_fluxes.jld2"))

animation_type = "Training"
n_trainings = length(train_files)
training_types = "Wind Mixing, Free Convection"
VIDEO_NAME = "test2"
animate_profiles_fluxes_comparison(plot_data, joinpath(OUTPUT_SOL_DIR, VIDEO_NAME), fps=30, 
                                                animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)
close(file)

FILE_NAME = "NDE_training_modified_pacanowski_philander_1sim_-1e-3_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_5e-3_rate_1e-4"
test_files = ["-1e-3", "-8e-4"]
timestep = 600
animate_training_results_oceananigans(test_files, timestep, FILE_NAME, OUTPUT_DIR)