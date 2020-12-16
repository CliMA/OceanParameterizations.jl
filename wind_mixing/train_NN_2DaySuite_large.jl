using Statistics
using NCDatasets
using Plots
using Flux
using OceanParameterizations
using Oceananigans.Grids
using BSON
using OrdinaryDiffEq, DiffEqSensitivity
include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")


# train_files = ["free_convection", "strong_wind", "weak_wind_strong_cooling"]
train_files = ["strong_wind"]
output_gif_directory = "Output"



𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    animate=false,
                    animate_dir="$(output_gif_directory)/Training")


function animate_NN(xs, y, t, x_str, x_label=["" for i in length(xs)], filename=x_str)
    PATH = joinpath(pwd(), "Output")
    anim = @animate for n in 1:size(xs[1],2)
    x_max = maximum(maximum(x) for x in xs)
    x_min = minimum(minimum(x) for x in xs)
        @info "$x_str frame of $n/$(size(xs[1],2))"
        fig = plot(xlim=(x_min, x_max), ylim=(minimum(y), maximum(y)), legend=:bottom)
        for i in 1:length(xs)
            plot!(fig, xs[i][:,n], y, label=x_label[i], title="t = $(round(t[n]/86400, digits=2)) days")
        end
        xlabel!(fig, "$x_str")
        ylabel!(fig, "z")
    end
    gif(anim, joinpath(PATH, "$(filename).gif"), fps=30)
end
# function append_parameters(𝒟, datanames)
#     filenames = Dict(
#         "free_convection"          => 1,
#         "strong_wind"              => 2,
#         "strong_wind_weak_cooling" => 3,
#         "weak_wind_strong_cooling" => 4,
#         "strong_wind_weak_heating" => 5,
#         "strong_wind_no_coriolis"  => 6,
#     )
#     # momentum_fluxes = [0., -1e-3, -8e-4, -3e-4, -1e-3, -2e-4]
#     # momentum_fluxes_scaling = ZeroMeanUnitVarianceScaling(momentum_fluxes)
#     # momentum_fluxes_scaled = scale(momentum_fluxes, momentum_fluxes_scaling)

#     # buoyancy_fluxes = [1.2e-7, 0., 3e-8, 1e-7, -4e-8, 0.]
#     # buoyancy_fluxes_scaling = ZeroMeanUnitVarianceScaling(buoyancy_fluxes)
#     # buoyancy_fluxes_scaled = scale(buoyancy_fluxes, buoyancy_fluxes_scaling)

#     fs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.]
#     fs_scaling = ZeroMeanUnitVarianceScaling(fs)
#     fs_scaled = scale(fs, fs_scaling)

#     datalength = Int(size(𝒟.uvT_scaled,2) / length(datanames))
#     # output = Array{Float64}(undef, size(𝒟.uvT_scaled,1)+3, size(𝒟.uvT_scaled,2))
#     output = Array{Float64}(undef, size(𝒟.uvT_scaled,1)+1, size(𝒟.uvT_scaled,2))
#     uvT = @view output[2:end, :]
#     uvT .= 𝒟.uvT_scaled


#     for i in 1:length(datanames)
#         dataname = datanames[i]
#         coriolis_row = @view output[1, (i-1)*datalength+1:i*datalength]
#         # momentum_row = @view output[2, (i-1)*datalength+1:i*datalength]
#         # buoyancy_row = @view output[3, (i-1)*datalength+1:i*datalength]
#         coriolis_row .= fs_scaled[filenames[dataname]]
#         # momentum_row .= momentum_fluxes_scaled[filenames[dataname]]
#         # buoyancy_row .= buoyancy_fluxes_scaled[filenames[dataname]]
#     end
#     return [(output[:,i], 𝒟.uw.scaled) for i in 1:size(output,2)], [(output[:,i], 𝒟.vw.scaled) for i in 1:size(output,2)], [(output[:,i], 𝒟.wT.scaled) for i in 1:size(output,2)]
# end

function prepare_training_data(input, truth)
    return [(input[:,i], truth[:,i]) for i in 1:size(truth, 2)]
end

uw_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.uw.scaled)
vw_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.vw.scaled)
wT_train = prepare_training_data(𝒟train.uvT_scaled, 𝒟train.wT.scaled)


N_inputs = 96
N_outputs = 31
uw_NN_model = Chain(Dense(N_inputs, 4*N_inputs, relu), Dense(4*N_inputs, 4*N_inputs, relu), Dense(4*N_inputs,N_outputs))
vw_NN_model = Chain(Dense(N_inputs, 4*N_inputs, relu), Dense(4*N_inputs, 4*N_inputs, relu), Dense(4*N_inputs,N_outputs))
wT_NN_model = Chain(Dense(N_inputs, 4*N_inputs, relu), Dense(4*N_inputs, 4*N_inputs, relu), Dense(4*N_inputs,N_outputs))

function predict(NN, x, y)
    interior = NN(x)
    return [y[1]; interior; y[end]]
end

predict(uw_NN_model, uw_train[1][1], uw_train[1][2])
Flux.Losses.mse(predict(uw_NN_model, uw_train[1][1], uw_train[1][2]), uw_train[1][2])
# uw_train[1][1]
# uw_NN_model(uw_train[1][1])

loss_uw(x, y) = Flux.Losses.mse(predict(uw_NN_model, x, y), y)
loss_vw(x, y) = Flux.Losses.mse(predict(vw_NN_model, x, y), y)
loss_wT(x, y) = Flux.Losses.mse(predict(wT_NN_model, x, y), y)

function train_NN(NN, loss, data, opts)
    function cb()
        @info "loss = $(mean([loss(data[i][1], data[i][2]) for i in 1:length(data)]))"
    end
   for opt in opts
        Flux.train!(loss, params(NN), data, opt, cb=Flux.throttle(cb, 2))
    end 
end

# optimizers = [ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), 
# Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent()]
# optimizers = [ADAM(), ADAM(), ADAM(), Descent(), Descent(), Descent()]
optimizers = [Descent()]


train_NN(uw_NN_model, loss_uw, uw_train, optimizers)
train_NN(vw_NN_model, loss_vw, vw_train, optimizers)
train_NN(wT_NN_model, loss_wT, wT_train, optimizers)

@info "loss = $(mean([loss_uw(uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train)]))"
@info "loss = $(mean([loss_vw(vw_train[i][1], vw_train[i][2]) for i in 1:length(vw_train)]))"
@info "loss = $(mean([loss_wT(wT_train[i][1], wT_train[i][2]) for i in 1:length(wT_train)]))"

PATH = pwd()
uw_NN_params = Dict(
    :neural_network => uw_NN_model)

bson(joinpath(PATH, "Output", "uw_NN_params_2DaySuite_large.bson"), uw_NN_params)

vw_NN_params = Dict(
    :neural_network => vw_NN_model)

bson(joinpath(PATH, "Output", "vw_NN_params_2DaySuite_large.bson"), vw_NN_params)


wT_NN_params = Dict(
    :neural_network => wT_NN_model)

bson(joinpath(PATH, "Output", "wT_NN_params_2DaySuite_large.bson"), wT_NN_params)


NN_prediction_uw = cat((predict(uw_NN_model, uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train))..., dims=2)
truth_uw = cat((uw_train[i][2] for i in 1:length(uw_train))..., dims=2)
uw_plots = (NN_prediction_uw, truth_uw)

NN_prediction_vw = cat((predict(vw_NN_model, vw_train[i][1], vw_train[i][2]) for i in 1:length(vw_train))..., dims=2)
truth_vw = cat((vw_train[i][2] for i in 1:length(vw_train))..., dims=2)
vw_plots = (NN_prediction_vw, truth_vw)


NN_prediction_wT = cat((predict(wT_NN_model, wT_train[i][1], wT_train[i][2]) for i in 1:length(wT_train))..., dims=2)
truth_wT = cat((wT_train[i][2] for i in 1:length(wT_train))..., dims=2)
wT_plots = (NN_prediction_wT, truth_wT)

animate_NN(uw_plots, 𝒟train.uw.z, 𝒟train.t[:,1], "uw", ["NN", "truth"], "uw_strong_wind_bounds1")
animate_NN(vw_plots, 𝒟train.vw.z, 𝒟train.t[:,1], "vw", ["NN", "truth"], "vw_strong_wind_bounds1")
animate_NN(wT_plots, 𝒟train.wT.z, 𝒟train.t[:,1], "wT", ["NN", "truth"], "wT_strong_wind_bounds")

𝒟train.uvT_scaled
𝒟train.uw.scaled