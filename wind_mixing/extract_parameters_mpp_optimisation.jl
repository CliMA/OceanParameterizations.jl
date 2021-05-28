using WindMixing
using Flux
using JLD2
using FileIO
using Plots

FILE_NAME = "parameter_optimisation_mpp_wind_mixing_-1e-3_cooling_5e-8"
FILE_PATH = joinpath(pwd(), "training_output", "$(FILE_NAME).jld2")
@assert isfile(FILE_PATH)
OUTPUT_PATH = joinpath(pwd(), "extracted_training_output", "$(FILE_NAME)_extracted.jld2")
# OUTPUT_PATH = joinpath("D:\\University Matters\\MIT\\CLiMA Project\\OceanParameterizations.jl\\training_output", "$(FILE_NAME)_extracted.jld2")

extract_parameters_modified_pacanowski_philander_optimisation(FILE_PATH, OUTPUT_PATH)

file = jldopen(OUTPUT_PATH)
@info file["parameters"]
losses = file["losses"]
close(file)

plot(1:length(losses), losses, yscale=:log10)
