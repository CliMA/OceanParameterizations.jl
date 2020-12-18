using BSON
using Flux

function export_NN(weights_str, NN_str, small_NDE=true)
    PATH = pwd()
    if small_NDE == true
        uw_NN = BSON.load(joinpath(PATH, "Output", "uw_NDE_1sim_100.bson"))[:neural_network]
        vw_NN = BSON.load(joinpath(PATH, "Output", "vw_NDE_1sim_100.bson"))[:neural_network]
        wT_NN = BSON.load(joinpath(PATH, "Output", "wT_NDE_1sim_100.bson"))[:neural_network]
    else
        uw_NN = BSON.load(joinpath(PATH, "Output", "uw_NN_params_2DaySuite_large.bson"))[:neural_network]
        vw_NN = BSON.load(joinpath(PATH, "Output", "vw_NN_params_2DaySuite_large.bson"))[:neural_network]
        wT_NN = BSON.load(joinpath(PATH, "Output", "wT_NN_params_2DaySuite_large.bson"))[:neural_network]
    end

    _, re_uw = Flux.destructure(uw_NN)
    _, re_vw = Flux.destructure(vw_NN)
    _, re_wT = Flux.destructure(wT_NN)

    uw_weights = BSON.load(joinpath(PATH, "Output", "$(weights_str[1]).bson"))[:weights]
    vw_weights = BSON.load(joinpath(PATH, "Output", "$(weights_str[2]).bson"))[:weights]
    wT_weights = BSON.load(joinpath(PATH, "Output", "$(weights_str[3]).bson"))[:weights]

    uw_NN = re_uw(uw_weights)
    vw_NN = re_vw(vw_weights)
    wT_NN = re_wT(wT_weights)

    bson(joinpath(PATH, "Output", "$(NN_str[1]).bson"), Dict(:neural_network => uw_NN))
    bson(joinpath(PATH, "Output", "$(NN_str[2]).bson"), Dict(:neural_network => vw_NN))
    bson(joinpath(PATH, "Output", "$(NN_str[3]).bson"), Dict(:neural_network => wT_NN))
end

weights_str = ["uw_NDE_weights_2DaySuite_2Sims", "vw_NDE_weights_2DaySuite_2Sims", "wT_NDE_weights_2DaySuite_2Sims"]
NN_str = ["uw_NDE_2sim_100", "vw_NDE_2sim_100", "wT_NDE_2sim_100"]

export_NN(weights_str, NN_str, true)