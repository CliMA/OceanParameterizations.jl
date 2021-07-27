using WindMixing
using OceanParameterizations
using OrdinaryDiffEq
using DiffEqSensitivity
using Flux
using GalacticOptim

train_files = [       
    # "wind_-5e-4_cooling_3e-8_new",   
    # "wind_-5e-4_cooling_2e-8_new",   
    # "wind_-5e-4_cooling_1e-8_new",   
    "wind_-3.5e-4_cooling_3e-8_new", 
    "wind_-3.5e-4_cooling_2e-8_new", 
    # "wind_-3.5e-4_cooling_1e-8_new", 
    # "wind_-2e-4_cooling_3e-8_new",   
    # "wind_-2e-4_cooling_2e-8_new",   
    # "wind_-2e-4_cooling_1e-8_new",   
    # "wind_-5e-4_heating_-3e-8_new",  
    # "wind_-5e-4_heating_-2e-8_new",  
    # "wind_-5e-4_heating_-1e-8_new",  
    # "wind_-3.5e-4_heating_-3e-8_new",
    # "wind_-3.5e-4_heating_-2e-8_new",
    # "wind_-3.5e-4_heating_-1e-8_new",
    # "wind_-2e-4_heating_-3e-8_new",  
    # "wind_-2e-4_heating_-2e-8_new",  
    # "wind_-2e-4_heating_-1e-8_new",  
]

PATH = pwd()
PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl"

FILE_NAME = "parameter_optimisation_18sim_windcooling_windheating_5params_LBFGS_scale_1e-3"
OUTPUT_PATH = joinpath(PATH, "training_output", "$(FILE_NAME).jld2")

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output", "$(FILE_NAME)_extracted.jld2")

timestepper = ROCK4()

optimizers = [LBFGS()]

tsteps = 1:20:1153
maxiters = 3

training_fractions = (T=0.8f0, profile=0.5f0, ∂T∂z=0.8f0)

optimise_modified_pacanowski_philander(train_files, tsteps, timestepper, optimizers, maxiters, OUTPUT_PATH, n_simulations=length(train_files),
                                       train_gradient=true, training_fractions=training_fractions)

extract_parameters_modified_pacanowski_philander_optimisation(OUTPUT_PATH, EXTRACTED_OUTPUT_PATH)