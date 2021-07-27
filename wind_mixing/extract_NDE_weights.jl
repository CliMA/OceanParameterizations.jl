using Flux
using JLD2

filename = "NDE_3sim_diurnal_18simBFGST0.8nograd_divide1f5_gradient_smallNN_leakyrelu_rate_2e-4_T0.8"

FILE_PATH = joinpath(pwd(), "extracted_training_output", "$(filename)_extracted.jld2")

OUTPUT_PATH = joinpath("D:\\Output_o", "$(filename)_weights.jld2")

file = jldopen(FILE_PATH)

uw_weights, _ = Flux.destructure(file["neural_network/uw"])
vw_weights, _ = Flux.destructure(file["neural_network/vw"])
wT_weights, _ = Flux.destructure(file["neural_network/wT"])

close(file)

output = jldopen(OUTPUT_PATH, "w") do file
    file["uw"] = uw_weights
    file["vw"] = vw_weights
    file["wT"] = wT_weights
end

