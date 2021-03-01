using Plots
using Flux
using WindMixing
using OceanParameterizations

train_files = ["strong_wind"]
uw_NN = BSON.load(joinpath(PATH, "NDEs", "uw_NN_large.bson"))[:neural_network]
vw_NN = BSON.load(joinpath(PATH, "NDEs", "vw_NN_large.bson"))[:neural_network]
wT_NN = BSON.load(joinpath(PATH, "NDEs", "wT_NN_large.bson"))[:neural_network]

𝒟train = data(train_files, scale_type=ZeroMeanUnitVarianceScaling, enforce_surface_fluxes=true)

function predict_NN(NN, x, y)
    interior = NN(x)
    return [y[1]; interior; y[end]]
end

function prepare_training_data(input, truth)
    return [(input[:,i], truth[:,i]) for i in 1:size(truth, 2)]
end

uw_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.uw.scaled)
vw_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.vw.scaled)
wT_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.wT.scaled)

NN_prediction_uw = cat((predict_NN(uw_NN, uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train))..., dims=2)
truth_uw = cat((uw_train[i][2] for i in 1:length(uw_train))..., dims=2)
uw_plots = (NN_prediction_uw, truth_uw)

NN_prediction_vw = cat((predict_NN(vw_NN, vw_train[i][1], vw_train[i][2]) for i in 1:length(vw_train))..., dims=2)
truth_vw = cat((vw_train[i][2] for i in 1:length(vw_train))..., dims=2)
vw_plots = (NN_prediction_vw, truth_vw)

NN_prediction_wT = cat((predict_NN(wT_NN, wT_train[i][1], wT_train[i][2]) for i in 1:length(wT_train))..., dims=2)
truth_wT = cat((wT_train[i][2] for i in 1:length(wT_train))..., dims=2)
wT_plots = (NN_prediction_wT, truth_wT)

animate_NN(uw_plots, 𝒟train.uw.z, 𝒟train.t[:,1], "uw", ["NN", "truth"], "uw_NN_SWNH_large")
animate_NN(vw_plots, 𝒟train.uw.z, 𝒟train.t[:,1], "vw", ["NN", "truth"], "vw_NN_SWNH_large")
animate_NN(wT_plots, 𝒟train.uw.z, 𝒟train.t[:,1], "wT", ["NN", "truth"], "wT_NN_SWNH_large")
