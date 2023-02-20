using WindMixing
using OceanParameterizations
using OrdinaryDiffEq
using DiffEqSensitivity
using Flux
using GalacticOptim
using LinearAlgebra
using Statistics
using Plots
using Oceananigans.Grids
using JLD2
using FileIO

BLAS.set_num_threads(1)

T_fraction = parse(Float32, ARGS[1])
N_sims = parse(Int, ARGS[2])
train_gradient = parse(Bool, ARGS[3])
optimizer_type = ARGS[4]

# rate_str = ARGS[4]
# rate = parse(Float64, rate_str)

# optimizer_type = "ADAM"
# T_fraction = 0.8f0
# N_sims = 1
# train_gradient = false
# optimizer_type = "BFGS"

train_files_all = [       
    "wind_-5e-4_cooling_3e-8_new",   
    "wind_-5e-4_cooling_1e-8_new",   
    "wind_-2e-4_cooling_3e-8_new",   
    "wind_-2e-4_cooling_1e-8_new",   
    "wind_-5e-4_heating_-3e-8_new",  
    "wind_-2e-4_heating_-1e-8_new",  
    "wind_-2e-4_heating_-3e-8_new",  
    "wind_-5e-4_heating_-1e-8_new",  

    "wind_-3.5e-4_cooling_2e-8_new", 
    "wind_-3.5e-4_heating_-2e-8_new",

    "wind_-5e-4_cooling_2e-8_new",   
    "wind_-3.5e-4_cooling_3e-8_new", 
    "wind_-3.5e-4_cooling_1e-8_new", 
    "wind_-2e-4_cooling_2e-8_new",   
    "wind_-3.5e-4_heating_-3e-8_new",
    "wind_-3.5e-4_heating_-1e-8_new",
    "wind_-2e-4_heating_-2e-8_new",  
    "wind_-5e-4_heating_-2e-8_new",  
]

train_files = train_files_all[1:N_sims]

PATH = pwd()
# PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl"
# PATH = "C:\\Users\\xinle\\MIT\\test"


if train_gradient
    FILE_NAME = "parameter_optimisation_$(N_sims)sim_windcooling_windheating_5params_$(optimizer_type)_T$(T_fraction)_nonu0Pr_var_grad_new"
else
    FILE_NAME = "parameter_optimisation_$(N_sims)sim_windcooling_windheating_5params_$(optimizer_type)_T$(T_fraction)_nonu0Pr_var_nograd_new"
end

OUTPUT_PATH = joinpath(PATH, "training_output", "$(FILE_NAME).jld2")
@assert !isfile(OUTPUT_PATH)

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output", "$(FILE_NAME)_extracted.jld2")

timestepper = ROCK4()

if optimizer_type == "LBFGS"
    optimizers = [LBFGS()]
else
    optimizers = [BFGS()]
end

# optimizers = [ADAM(rate)]

tsteps = 1:20:1153
maxiters = 200
# maxiters = 10

training_fractions = (T=T_fraction, profile=0.5f0, ∂T∂z=T_fraction)

optimise_modified_pacanowski_philander_noν₀Pr(train_files, tsteps, timestepper, optimizers, maxiters, OUTPUT_PATH, n_simulations=length(train_files),
                                       train_gradient=train_gradient, training_fractions=training_fractions)
                                    #    ν₀ = 1f-5, ν₋ = 5.77f-1, ΔRi=2.65f-1, Riᶜ=-2.71f-1, Pr=1.65f0)

extract_parameters_modified_pacanowski_philander_optimisation(OUTPUT_PATH, EXTRACTED_OUTPUT_PATH)

PATH = joinpath(pwd(), "extracted_training_output")

DATA_NAME = FILE_NAME

DATA_PATH = joinpath(PATH, "$(DATA_NAME)_extracted.jld2")
ispath(DATA_PATH)

FILE_PATH = joinpath(pwd(), "Output", DATA_NAME)

if !ispath(FILE_PATH)
    mkdir(FILE_PATH)
end


file = jldopen(DATA_PATH, "r")

train_files = file["training_info/train_files"]
train_parameters = file["training_info/parameters"]
loss_scalings = file["training_info/loss_scalings"]
mpp_parameters = file["parameters"]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
close(file)

ν₋, ΔRi, Riᶜ = mpp_parameters
ν₀ = 1f-5
Pr = 1f0

N_inputs = 96
hidden_units = 400
N_outputs = 31

weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

uw_NN = re(zeros(Float32, length(weights)))
vw_NN = re(zeros(Float32, length(weights)))
wT_NN = re(zeros(Float32, length(weights)))

to_run = [
    "wind_-5e-4_cooling_3e-8_new",   
    "wind_-5e-4_cooling_1e-8_new",   
    "wind_-2e-4_cooling_3e-8_new",   
    "wind_-2e-4_cooling_1e-8_new",   
    "wind_-5e-4_heating_-3e-8_new",  
    "wind_-2e-4_heating_-1e-8_new",  
    "wind_-2e-4_heating_-3e-8_new",  
    "wind_-5e-4_heating_-1e-8_new",  
  
    "wind_-3.5e-4_cooling_2e-8_new", 
    "wind_-3.5e-4_heating_-2e-8_new",
  
    "wind_-5e-4_cooling_2e-8_new",   
    "wind_-3.5e-4_cooling_3e-8_new", 
    "wind_-3.5e-4_cooling_1e-8_new", 
    "wind_-2e-4_cooling_2e-8_new",   
    "wind_-3.5e-4_heating_-3e-8_new",
    "wind_-3.5e-4_heating_-1e-8_new",
    "wind_-2e-4_heating_-2e-8_new",  
    "wind_-5e-4_heating_-2e-8_new",  
  ]

for test_file in to_run
    @info "running $test_file"
    test_files = [test_file]
    𝒟test = WindMixing.data(test_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=false)
    trange = 1:1:1153
    plot_data = NDE_profile(uw_NN, vw_NN, wT_NN, test_file, 𝒟test, 𝒟train, trange,
                            modified_pacanowski_philander=true, 
                            ν₀=ν₀, ν₋=ν₋, ΔRi=ΔRi, Riᶜ=Riᶜ, Pr=Pr,
                            convective_adjustment=false,
                            smooth_NN=false, smooth_Ri=false,
                            zero_weights=true,
                            loss_scalings=loss_scalings)

    animation_type = "Pre-Training"
    n_trainings = length(train_files)
    training_types = "Modified Pacanowski-Philander"
    VIDEO_NAME = "$(test_file)_mpp_nonu0"
    animate_profiles_fluxes_comparison(plot_data, plot_data, plot_data, joinpath(FILE_PATH, VIDEO_NAME), fps=30, 
                                                    animation_type=animation_type, n_trainings=n_trainings, training_types=training_types)
end
