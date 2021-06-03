using Flux
using WindMixing
using JLD2
using FileIO
using OceanParameterizations
using OrdinaryDiffEq
using Random
using GalacticOptim
using LinearAlgebra
using WindMixing: solve_NDE_nonmutating
using WindMixing: solve_NDE_nonmutating_backprop
using WindMixing: calculate_profile_gradient
using BenchmarkTools

train_files = ["-1e-3"]
# train_files = [
#                "wind_-1e-3_cooling_4e-8", 
#             #    "wind_-2e-4_cooling_1e-8", 
#             #    "wind_-1e-3_cooling_2e-8", 
#                "wind_-2e-4_cooling_5e-8", 
#                "wind_-5e-4_cooling_3e-8"
#                ]

𝒟train = WindMixing.data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

PATH = pwd()
FILE_PATH_NN = joinpath(PATH, "extracted_training_output", 
                        "NDE_training_modified_pacanowski_philander_1sim_-1e-3_diffusivity_1e-1_Ri_1e-1_weights_divide1f5_gradient_smallNN_scale_1e-2_rate_1e-4_extracted.jld2")

@assert isfile(FILE_PATH_NN)
file = jldopen(FILE_PATH_NN, "r")

uw_NN = file["neural_network/uw"]
vw_NN = file["neural_network/vw"]
wT_NN = file["neural_network/wT"]
close(file)

# N_inputs = 96
# hidden_units = 400
# N_outputs = 31

# # weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs)))
# weights, re = Flux.destructure(Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, N_outputs)))

# uw_NN = re(weights ./ 1f5)
# vw_NN = re(weights ./ 1f5)
# wT_NN = re(weights ./ 1f5)

tsteps = 1:25:1153
timestepper = ROCK4()
n_simulations = length(train_files)
solve_NDE_nonmutating(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper; 
                                n_simulations=n_simulations, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=1.67f-4, g=9.81f0)

optimizer = ADAM(2e-4)
maxiters = 3
solve_NDE_nonmutating_backprop(uw_NN, vw_NN, wT_NN, 𝒟train, tsteps, timestepper, optimizer, 
                                maxiters=maxiters, n_simulations=n_simulations, gradient_scaling=1f-2, ν₀=1f-4, ν₋=1f-1, ΔRi=1f0, Riᶜ=0.25, Pr=1f0, κ=10f0, f=1f-4, α=1.67f-4, g=9.81f0)