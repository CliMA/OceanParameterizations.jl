using WindMixing
using OceanParameterizations
using OrdinaryDiffEq
using DiffEqSensitivity
using Flux
using GalacticOptim

train_files = [       
    "wind_-1e-3_cooling_4e-8",
    "wind_-1e-3_cooling_2e-8",
    "wind_-2e-4_cooling_5e-8",
    "wind_-1e-3_heating_-4e-8",
    "wind_-2e-4_heating_-5e-8",
]

PATH = pwd()
FILE_NAME = "parameter_optimisation_5sim_windcooling_SS_SW_WS_windheating_SS_WS"
OUTPUT_PATH = joinpath(PATH, "training_output", "$(FILE_NAME).jld2")

EXTRACTED_OUTPUT_PATH = joinpath(PATH, "extracted_training_output", "$(FILE_NAME)_extracted.jld2")

timestepper = ROCK4()

optimizers = [LBFGS()]

tsteps = 1:20:1153
maxiters = 200
results, parameters = optimise_modified_pacanowski_philander(train_files, tsteps, timestepper, optimizers, maxiters, FILE_PATH, n_simulations=length(train_files))

extract_parameters_modified_pacanowski_philander_optimisation(OUTPUT_PATH, EXTRACTED_OUTPUT_PATH)