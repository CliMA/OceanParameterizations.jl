using WindMixing
using OceanParameterizations
using OrdinaryDiffEq
using DiffEqSensitivity
using Flux

train_files = ["-1e-3", "cooling_5e-8"]

OUTPUT_PATH = joinpath(pwd(), "extracted_training_output")
# OUTPUT_PATH = "D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output"
FILE_PATH = joinpath(OUTPUT_PATH, "parameter_optimisation_mpp_wind_mixing_-1e-3_cooling_5e-8.jld2")

# 𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)
timestepper = ROCK4()

optimizers = [ADAM(2e-4)]

tsteps = 1:25:1153
maxiters = 600
parameters = optimise_modified_pacanowski_philander(train_files, tsteps, timestepper, optimizers, maxiters, FILE_PATH, n_simulations=length(train_files))