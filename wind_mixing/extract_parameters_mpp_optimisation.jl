using WindMixing
using Flux
using JLD2
using FileIO
using Plots

FILE_NAME = "parameter_optimisation_18sim_windcooling_windheating_5params_BFGS"
FILE_PATH = joinpath(pwd(), "training_output", "$(FILE_NAME).jld2")
@assert isfile(FILE_PATH)
OUTPUT_PATH = joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2")
OUTPUT_PATH = joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output", "$(FILE_NAME)_extracted.jld2")

extract_parameters_modified_pacanowski_philander_optimisation(FILE_PATH, OUTPUT_PATH)

file = jldopen(OUTPUT_PATH)
@info file["parameters"]
losses = file["losses"]

ν₀_initial = 1f-4
ν₋_initial = 1f-1
ΔRi_initial = 1f-1
Riᶜ_initial = 0.25f0
Pr_initial = 1f0

scalings = 1 ./ [ν₀_initial, ν₋_initial, ΔRi_initial, Riᶜ_initial, Pr_initial]

@info file["parameters"] ./ scalings

close(file)

plot(1:length(losses), losses, yscale=:log10)
