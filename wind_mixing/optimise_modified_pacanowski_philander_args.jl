using WindMixing
using OceanParameterizations
using OrdinaryDiffEq
using DiffEqSensitivity
using Flux
using GalacticOptim
using LinearAlgebra

BLAS.set_num_threads(1)

T_fraction = parse(Float32, ARGS[1])
N_sims = parse(Int, ARGS[2])
train_gradient = parse(Bool, ARGS[3])
optimizer_type = ARGS[4]

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

if train_gradient
    FILE_NAME = "parameter_optimisation_$(N_sims)sim_windcooling_windheating_5params_$(optimizer_type)_T$(T_fraction)_grad"
else
    FILE_NAME = "parameter_optimisation_$(N_sims)sim_windcooling_windheating_5params_$(optimizer_type)_T$(T_fraction)_nograd"
end

OUTPUT_PATH = joinpath(PATH, "training_output", "$(FILE_NAME).jld2")

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output", "$(FILE_NAME)_extracted.jld2")

timestepper = ROCK4()

if optimizer_type == "LBFGS"
    optimizers = [LBFGS()]
else
    optimizers = [BFGS()]
end

tsteps = 1:20:1153
maxiters = 200

training_fractions = (T=T_fraction, profile=0.5f0, ∂T∂z=T_fraction)

optimise_modified_pacanowski_philander(train_files, tsteps, timestepper, optimizers, maxiters, OUTPUT_PATH, n_simulations=length(train_files),
                                       train_gradient=train_gradient, training_fractions=training_fractions)

extract_parameters_modified_pacanowski_philander_optimisation(OUTPUT_PATH, EXTRACTED_OUTPUT_PATH)