using JLD2
using FileIO
using Flux
using OrdinaryDiffEq, DiffEqSensitivity

FILE_PATH = joinpath(pwd(), "training_output", "NDE_training_1sim_convective_adjustment.jld2")
OUTPUT_PATH = joinpath(pwd(), "training_output", "NDE_training_1sim_convective_adjustment_temp3.jld2")

@info "Opening file"
file = jldopen(FILE_PATH, "r")

N_stages = length(keys(file["training_data/loss"]))

@info "Loading file"
# training_info = file["training_info"]
uw_NNs = file["training_data/neural_network/uw/$(N_stages-1)"]
vw_NNs = file["training_data/neural_network/vw/$(N_stages-1)"]
wT_NNs = file["training_data/neural_network/wT/$(N_stages-1)"]

N_data = length(keys(uw_NNs))
losses = file["training_data/loss/$(N_stages-1)"]

output_size = 200

@info "Writing file"
jldopen(OUTPUT_PATH, "w") do file
    # @info "Writing Training Info"
    # for key in keys(training_info)
    #     file["training_info/$key"] = training_info[key]
    # end

    for i in N_data - output_size + 1:1:N_data
        @info "Writing NN $i/$N_data"
        file["neural_network/uw/$i"] = uw_NNs["$i"]
        file["neural_network/vw/$i"] = vw_NNs["$i"]
        file["neural_network/wT/$i"] = wT_NNs["$i"]
    end

    for i in N_data - output_size + 1:1:N_data
        @info "Writing loss $i/$N_data"
        file["loss/$i"] = losses["$i"]
    end

end

@info "End"