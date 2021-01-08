using BSON
using Flux

# Export the Neural Network given their weights
function export_NN(weights_str, NN_str, uw_NN, vw_NN, wT_NN)
    PATH = pwd()

    _, re_uw = Flux.destructure(uw_NN)
    _, re_vw = Flux.destructure(vw_NN)
    _, re_wT = Flux.destructure(wT_NN)

    uw_weights = BSON.load(joinpath(PATH, "Output", "$(weights_str[1]).bson"))[:weights]
    vw_weights = BSON.load(joinpath(PATH, "Output", "$(weights_str[2]).bson"))[:weights]
    wT_weights = BSON.load(joinpath(PATH, "Output", "$(weights_str[3]).bson"))[:weights]

    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)

    bson(joinpath(PATH, "NDEs", "$(NN_str[1]).bson"), Dict(:neural_network => uw_NN))
    bson(joinpath(PATH, "NDEs", "$(NN_str[2]).bson"), Dict(:neural_network => vw_NN))
    bson(joinpath(PATH, "NDEs", "$(NN_str[3]).bson"), Dict(:neural_network => wT_NN))
end

# uw_NN = BSON.load(joinpath(PATH, "NDEs", "uw_NDE_SWNH_100.bson"))[:neural_network]
# vw_NN = BSON.load(joinpath(PATH, "NDEs", "vw_NDE_SWNH_100.bson"))[:neural_network]
# wT_NN = BSON.load(joinpath(PATH, "NDEs", "wT_NDE_SWNH_100.bson"))[:neural_network]

N_inputs = 96
hidden_units = 400
N_outputs = 31
uw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
vw_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))
wT_NN = Chain(Dense(N_inputs, hidden_units, relu), Dense(hidden_units, hidden_units, relu), Dense(hidden_units, N_outputs))


# File name of the weights
weights_str = ["uw_weights_large_NN", "vw_weights_large_NN", "wT_weights_large_NN"]

# Names of the output files
NN_str = ["uw_NN_large", "vw_NN_large", "wT_NN_large"]

export_NN(weights_str, NN_str, uw_NN, vw_NN, wT_NN)