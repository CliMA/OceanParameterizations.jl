using Random
using Statistics
using Printf
using Logging

using LoggingExtras
using DataDeps
using GeoData
using Flux
using JLD2

using OceanParameterizations
using FreeConvection

using Oceananigans: OceananigansLogger
using FreeConvection: coarse_grain

ENV["GKSwstype"] = "100"

output_dir = joinpath(@__DIR__, "layers3_depth4_relu")
mkpath(output_dir)

log_filepath = joinpath(output_dir, "layers3_depth4_relu.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger

## Neural differential equation parameters

Nz = 32  # Number of grid points (to coarse grain to)

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

@info "Loading training data..."
training_datasets = tds = Dict{Int,Any}(
    1 => NCDstack(datadep"lesbrary_free_convection_1/statistics.nc"),
    2 => NCDstack(datadep"lesbrary_free_convection_2/statistics.nc")
)

## Add surface fluxes to data

@info "Inserting surface fluxes..."
training_datasets = tds = Dict{Int,Any}(id => add_surface_fluxes(ds) for (id, ds) in tds)

## Coarse grain training data

@info "Coarse graining training data..."
coarse_training_datasets = ctds =
    Dict{Int,Any}(id => coarse_grain(ds, Nz) for (id, ds) in tds)

## Create animations for T(z,t) and wT(z,t)

@info "Animating training data..."
for id in keys(tds)
    T_filepath = joinpath(output_dir, "free_convection_T_$id")
    animate_variable(tds[id][:T], ctds[id][:T], xlabel="Temperature T (°C)", filepath=T_filepath, frameskip=5)

    wT_filepath = joinpath(output_dir, "free_convection_wT_$id")
    animate_variable(tds[id][:wT], ctds[id][:wT], xlabel="Heat flux wT (m/s °C)", filepath=wT_filepath, frameskip=5)
end

## Pull out input (T) and output (wT) training data

@info "Wrangling training data..."
input_training_data = wrangle_input_training_data(coarse_training_datasets)
output_training_data = wrangle_output_training_data(coarse_training_datasets)

## Feature scaling

@info "Scaling features..."

T_training_data = cat([input.temperature for input in input_training_data]..., dims=2)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

input_training_data = [rescale(i, T_scaling, wT_scaling) for i in input_training_data]
output_training_data = wT_scaling.(output_training_data)

## Training data pairs

@info "Batching training data..."
n_training_data = length(input_training_data)
training_data = [(input_training_data[n], output_training_data[:, n]) for n in 1:n_training_data] |> shuffle
data_loader = Flux.Data.DataLoader(training_data, batchsize=n_training_data, shuffle=true)

n_obs = data_loader.nobs
batch_size = data_loader.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Training data loader contains $n_obs pairs of observations (batch size = $batch_size)."

## Train neural network on T -> wT mapping

@info "Training neural network..."

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

@info "Animating neural network's heat flux predictions..."
for (id, ds) in coarse_training_datasets
    filepath = joinpath(output_dir, "learned_heat_flux_initial_guess_$id")
    animate_learned_heat_flux(ds, free_convection_neural_network, T_scaling, wT_scaling, filepath=filepath)
end

## Save neural network + weights

initial_nn_filepath = joinpath(output_dir, "free_convection_initial_neural_network.jld2")
@info "Saving initial neural network to $initial_nn_filepath..."
jldopen(initial_nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end

## Train neural differential equation on incrementally increasing time spans

nn_history_filepath = joinpath(output_dir, "neural_network_history.jld2")

training_iterations = (1:20, 1:2:101, 1:4:201)
training_epochs     = (20,   20,      20)
opt = ADAM()

for (iterations, epochs) in zip(training_iterations, training_epochs)
    @info "Training free convection NDE with iterations=$iterations for $epochs epochs  with $(typeof(opt))(η=$(opt.eta))..."
    train_neural_differential_equation!(NN, coarse_training_datasets, T_scaling, wT_scaling, iterations, opt,
                                        epochs, history_filepath=nn_history_filepath)
end

## Train on entire solution while decreasing the learning rate

burn_in_iterations = 1:4:289
burn_in_epochs = 50
optimizers = [ADAM(1e-3), ADAM(1e-4)]

for opt in optimizers
    @info "Training free convection NDE with iterations=$burn_in_iterations for $burn_in_epochs epochs  with $(typeof(opt))(η=$(opt.eta))..."
    train_neural_differential_equation!(NN, coarse_training_datasets, T_scaling, wT_scaling, burn_in_iterations, opt,
                                        burn_in_epochs, history_filepath=nn_history_filepath)
end

## Save neural network + weights

final_nn_filepath = joinpath(output_dir, "free_convection_final_neural_network.jld2")
@info "Saving final neural network to $final_nn_filepath..."
jldopen(final_nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end
