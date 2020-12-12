using Random
using Statistics
using Printf
using DataDeps
using GeoData
using Flux
using BSON

using OceanParameterizations
using FreeConvection
using FreeConvection: coarse_grain

## Neural differential equation parameters

Nz = 32

NN = Chain(Dense( Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

function free_convection_neural_network(input)
    ϕ = NN(input.temperature)
    wT = cat(input.bottom_flux, ϕ, input.top_flux, dims=1)
    return wT
end

## Register data dependencies

@info "Registering data dependencies..."

for dd in FreeConvection.LESBRARY_DATA_DEPS
    DataDeps.register(dd)
end

## Load training data

training_datasets = tds = Dict(
    1 => NCDstack(datadep"lesbrary_free_convection_1/statistics.nc"),
    2 => NCDstack(datadep"lesbrary_free_convection_2/statistics.nc")
)

## Add surface fluxes to data

training_datasets = tds = Dict(id => add_surface_fluxes(ds) for (id, ds) in tds)

## Coarse grain training data

coarse_training_datasets = ctds =
    Dict(id => coarse_grain(ds, Nz) for (id, ds) in tds)

## Create animations for T(z,t) and wT(z,t)

for id in keys(tds)
    T_filepath = "free_convection_T_$id"
    animate_variable(tds[id][:T], ctds[id][:T], xlabel="Temperature T (°C)", filepath=T_filepath, frameskip=5)

    wT_filepath = "free_convection_wT_$id"
    animate_variable(tds[id][:wT], ctds[id][:wT], xlabel="Heat flux wT (m/s °C)", filepath=wT_filepath, frameskip=5)
end

## Pull out input (T) and output (wT) training data

input_training_data = wrangle_input_training_data(coarse_training_datasets)
output_training_data = wrangle_output_training_data(coarse_training_datasets)

## Feature scaling

T_training_data = cat([input.temperature for input in input_training_data]..., dims=2)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

input_training_data = [rescale(i, T_scaling, wT_scaling) for i in input_training_data]
output_training_data = wT_scaling.(output_training_data)

## Training data pairs

n_training_data = length(input_training_data)
training_data = [(input_training_data[n], output_training_data[:, n]) for n in 1:n_training_data] |> shuffle
data_loader = Flux.Data.DataLoader(training_data, batchsize=n_training_data, shuffle=true)

n_obs = data_loader.nobs
batch_size = data_loader.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Training data loader contains $n_obs pairs of observations (batch size = $batch_size)."

## Train neural network on T -> wT mapping

nn_loss(input, output) = Flux.mse(free_convection_neural_network(input), output)

nn_training_set_loss(training_data) = mean(nn_loss(input, output) for (input, output) in training_data)

function nn_callback()
    μ_loss = nn_training_set_loss(training_data)
    @info @sprintf("Training free convection neural network... training set MSE loss = %.12e", μ_loss)
    return μ_loss
end

epochs = 2
optimizers = [ADAM(1e-3), Descent(1e-4)]

for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader)
    @info "Training heat flux neural network with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
    Flux.train!(nn_loss, Flux.params(NN), mini_batch, opt, cb=Flux.throttle(nn_callback, 5))
end

## Animate the heat flux the neural network has learned

for (id, ds) in coarse_training_datasets
    filepath = "learned_heat_flux_initial_guess_$id"
    animate_learned_heat_flux(ds, free_convection_neural_network, T_scaling, wT_scaling, filepath=filepath)
end

## Save neural network + weights

neural_network_parameters = Dict(
       :grid_points => Nz,
    :neural_network => NN,
         :T_scaling => T_scaling,
        :wT_scaling => wT_scaling)

bson("free_convection_neural_network_parameters.bson", neural_network_parameters)
