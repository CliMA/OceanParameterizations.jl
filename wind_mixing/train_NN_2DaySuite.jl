using Statistics
using NCDatasets
using Plots
using Flux, DiffEqFlux
using OceanParameterizations
using Oceananigans.Grids
using BSON
include("lesbrary_data.jl")
include("data_containers.jl")
include("animate_prediction.jl")


train_files = ["free_convection", "strong_wind", "strong_wind_no_coriolis"]
output_gif_directory = "Output"



𝒟train = data(train_files,
                    scale_type=ZeroMeanUnitVarianceScaling,
                    animate=false,
                    animate_dir="$(output_gif_directory)/Training")


function append_parameters(𝒟, datanames)
    filenames = Dict(
        "free_convection"          => 1,
        "strong_wind"              => 2,
        "strong_wind_weak_cooling" => 3,
        "weak_wind_strong_cooling" => 4,
        "strong_wind_weak_heating" => 5,
        "strong_wind_no_coriolis"  => 6,
    )
    momentum_fluxes = [0., -1e-3, -8e-4, -3e-4, -1e-3, -2e-4]
    momentum_fluxes_scaling = ZeroMeanUnitVarianceScaling(momentum_fluxes)
    momentum_fluxes_scaled = scale(momentum_fluxes, momentum_fluxes_scaling)

    buoyancy_fluxes = [1.2e-7, 0., 3e-8, 1e-7, -4e-8, 0.]
    buoyancy_fluxes_scaling = ZeroMeanUnitVarianceScaling(buoyancy_fluxes)
    buoyancy_fluxes_scaled = scale(buoyancy_fluxes, buoyancy_fluxes_scaling)

    fs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.]
    fs_scaling = ZeroMeanUnitVarianceScaling(fs)
    fs_scaled = scale(fs, fs_scaling)

    datalength = Int(size(𝒟.uvT_scaled,2) / length(datanames))
    output = Array{Float64}(undef, size(𝒟.uvT_scaled,1)+3, size(𝒟.uvT_scaled,2))
    uvT = @view output[4:end, :]
    uvT .= 𝒟.uvT_scaled


    for i in 1:length(datanames)
        dataname = datanames[i]
        coriolis_row = @view output[1, (i-1)*datalength+1:i*datalength]
        momentum_row = @view output[2, (i-1)*datalength+1:i*datalength]
        buoyancy_row = @view output[3, (i-1)*datalength+1:i*datalength]
        coriolis_row .= fs_scaled[filenames[dataname]]
        momentum_row .= momentum_fluxes_scaled[filenames[dataname]]
        buoyancy_row .= buoyancy_fluxes_scaled[filenames[dataname]]
    end
    return [(output[:,i], 𝒟.uw.scaled) for i in 1:size(output,2)], [(output[:,i], 𝒟.vw.scaled) for i in 1:size(output,2)], [(output[:,i], 𝒟.wT.scaled) for i in 1:size(output,2)]
end

uw_train, vw_train, wT_train = append_parameters(𝒟train, train_files)

uw_NN_model = Chain(Dense(99,99, relu), Dense(99,99, relu), Dense(99,33))
vw_NN_model = Chain(Dense(99,99, relu), Dense(99,99, relu), Dense(99,33))
wT_NN_model = Chain(Dense(99,99, relu), Dense(99,99, relu), Dense(99,33))

loss_uw(x, y) = Flux.Losses.mse(uw_NN_model(x), y)
loss_vw(x, y) = Flux.Losses.mse(vw_NN_model(x), y)
loss_wT(x, y) = Flux.Losses.mse(wT_NN_model(x), y)

function train_NN(NN, loss, data, opts)
    function cb()
        @info "loss = $(mean([loss(data[i][1], data[i][2]) for i in 1:length(data)]))"
    end
   for opt in opts
        Flux.train!(loss, params(NN), data, opt, cb=Flux.throttle(cb, 2))
    end 
end

optimizers = [ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), ADAM(), 
Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent(), Descent()]

train_NN(uw_NN_model, loss_uw, uw_train, optimizers)

# @info "loss = $(mean([loss_uw(uw_train[i][1], uw_train[i][2]) for i in 1:length(uw_train)]))"

uw_NN_params = Dict(
    :neural_network => uw_NN_model)

bson("uw_NN_params_2DaySuite.bson", uw_NN_params)

train_NN(vw_NN_model, loss_vw, vw_train, optimizers)

vw_NN_params = Dict(
    :neural_network => vw_NN_model)

bson("vw_NN_params_2DaySuite.bson", vw_NN_params)

train_NN(wT_NN_model, loss_wT, wT_train, optimizers)
wT_NN_params = Dict(
    :neural_network => wT_NN_model)

bson("wT_NN_params_2DaySuite.bson", wT_NN_params)
