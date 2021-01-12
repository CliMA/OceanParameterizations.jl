using JLD2
using FileIO
using Flux
using OrdinaryDiffEq, DiffEqSensitivity

FILE_PATH = joinpath(pwd(), "training_output", "NDE_training_1sim_convective_adjustment.jld2")
OUTPUT_PATH = joinpath(pwd(), "training_output", "NDE_training_1sim_convective_adjustment_temp.jld2")

@info "Opening file"
file = jldopen(FILE_PATH, "r")

N_stages = length(keys(file["training_data/loss"]))

@info "Loading file"
training_info = file["training_info"]
uw_NNs = file["training_data/neural_network/uw/$(N_stages-1)"]
vw_NNs = file["training_data/neural_network/vw/$(N_stages-1)"]
wT_NNs = file["training_data/neural_network/wT/$(N_stages-1)"]

losses = file["training_data/loss/$(N_stages-1)"]

@info "Writing file"
jldopen(OUTPUT_PATH, "w") do file
    # @info "Writing Training Info"
    # for key in keys(training_info)
    #     file["training_info/$key"] = training_info[key]
    # end

    @info "Writing NNs"
    for key in keys(uw_NNs)
        file["neural_network/uw/$key"] = uw_NNs[key]
        file["neural_network/vw/$key"] = vw_NNs[key]
        file["neural_network/wT/$key"] = wT_NNs[key]
    end

    @info "Writing losses"

    for key in keys(losses)
        file["loss/$key"] = losses[key]
    end

end

@info "End"