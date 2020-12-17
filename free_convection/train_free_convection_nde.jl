using Random
using Statistics
using Printf
using Logging

using ArgParse
using LoggingExtras
using DataDeps
using GeoData
using Flux
using JLD2

using OceanParameterizations
using FreeConvection

using Oceananigans: OceananigansLogger
using FreeConvection: coarse_grain

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
ENV["GKSwstype"] = "100"

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--grid-points"
            help = "Number of vertical grid points in the trained neural differential equation (LES data will be coarse-grained to this resolution)."
            default = 32
            arg_type = Int

        "--nde"
            help = "Type of neural differential equation (NDE) to train. Options: free_convection, convective_adjustment"
            default = "convective_adjustment"
            arg_type = AbstractString

        "--name"
            help = "Experiment name (also determines name of output directory)."
            default = "layers3_depth4_relu"
            arg_type = AbstractString
    end

    return parse_args(settings)
end

## Parse command line arguments

@info "Parsing command line arguments..."
args = parse_command_line_arguments()

experiment_name = args["name"]

output_dir = joinpath(@__DIR__, experiment_name)
mkpath(output_dir)

log_filepath = joinpath(output_dir, "$experiment_name.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger

nde_type = Dict(
    "free_convection" => FreeConvectionNDE,
    "convective_adjustment" => ConvectiveAdjustmentNDE
)

NDEType = nde_type[args["nde"]]

## Neural network architecture

Nz = args["grid-points"]

NN = Chain(Dense( Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end

## Register data dependencies

@info "Registering data dependencies..."
for dd in FreeConvection.LESBRARY_DATA_DEPS
    DataDeps.register(dd)
end

## Load data

@info "Loading data..."
datasets = Dict{Int,Any}(
    1 => NCDstack(datadep"free_convection_8days_Qb1e-8/statistics.nc"),
    2 => NCDstack(datadep"free_convection_8days_Qb2e-8/statistics.nc"),
    3 => NCDstack(datadep"free_convection_8days_Qb3e-8/statistics.nc"),
    4 => NCDstack(datadep"free_convection_8days_Qb4e-8/statistics.nc"),
    5 => NCDstack(datadep"free_convection_8days_Qb5e-8/statistics.nc"),
    6 => NCDstack(datadep"free_convection_8days_Qb6e-8/statistics.nc")
)

## Add surface fluxes to data

@info "Inserting surface fluxes..."
datasets = Dict{Int,Any}(id => add_surface_fluxes(ds) for (id, ds) in datasets)

## Coarse grain training data

@info "Coarse graining data..."
coarse_datasets = Dict{Int,Any}(id => coarse_grain(ds, Nz) for (id, ds) in datasets)

## Split into training and testing data

@info "Partitioning data into training and testing datasets..."

ids_train = [1, 2, 4, 6]
ids_test = [3, 5]

training_datasets = Dict(id => datasets[id] for id in ids_train)
testing_datasets = Dict(id => datasets[id] for id in ids_test)

coarse_training_datasets = Dict(id => coarse_datasets[id] for id in ids_train)
coarse_testing_datasets = Dict(id => coarse_datasets[id] for id in ids_test)

## Create animations for T(z,t) and wT(z,t)

@info "Animating training data..."
for id in keys(training_datasets)
    filepath = joinpath(output_dir, "free_convection_data_$id")
    animate_data(training_datasets[id], coarse_training_datasets[id]; filepath, frameskip=5)
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

epochs = 5
optimizers = [ADAM(1e-3), Descent(1e-4)]

for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader)
    @info "Training heat flux neural network with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
    Flux.train!(nn_loss, Flux.params(NN), mini_batch, opt, cb=Flux.throttle(nn_callback, 5))
end

## Animate the heat flux the neural network has learned

@info "Animating what the neural network has learned..."
for (id, ds) in coarse_training_datasets
    filepath = joinpath(output_dir, "learned_free_convection_initial_guess_$id")
    animate_learned_free_convection(ds, NN, free_convection_neural_network, NDEType, T_scaling, wT_scaling, filepath=filepath)
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

training_iterations = (1:20, 1:5:101, 1:10:201, 1:20:401, 1:40:801)
training_epochs     = (20,   20,      20,       20,       20)
opt = ADAM()

for (iterations, epochs) in zip(training_iterations, training_epochs)
    @info "Training free convection NDE with iterations=$iterations for $epochs epochs  with $(typeof(opt))(η=$(opt.eta))..."
    train_neural_differential_equation!(NN, NDEType, coarse_training_datasets, T_scaling, wT_scaling, iterations, opt,
                                        epochs, history_filepath=nn_history_filepath)
end

## Train on entire solution while decreasing the learning rate

burn_in_iterations = 1:32:1153
burn_in_epochs = 200
optimizers = [ADAM(1e-3)]

for opt in optimizers
    @info "Training free convection NDE with iterations=$burn_in_iterations for $burn_in_epochs epochs with $(typeof(opt))(η=$(opt.eta))..."
    train_neural_differential_equation!(NN, NDEType, coarse_training_datasets, T_scaling, wT_scaling, burn_in_iterations, opt,
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

## Analyze NDE accuracy

true_solutions = Dict(id => T_scaling.(ds[:T].data) for (id, ds) in coarse_datasets)
nde_solution_history = compute_nde_solution_history(coarse_datasets, final_nn_filepath, nn_history_filepath)

for id in (ids_train..., ids_test...)
    @info @sprintf("MDE loss (id=%s): %.12e", id, Flux.mse(true_solutions[id], nde_solution_history[id][end]))
end

plot_epoch_loss(ids_train, ids_test, nde_solution_history, true_solutions,
                title = "Free convection NDE",
                filepath = joinpath(output_dir, "free_convection_nde_loss_history.png"))

animate_nde_loss(coarse_datasets, ids_train, ids_test, nde_solution_history, true_solutions,
                 title = "Free convection NDE",
                 filepath = joinpath(output_dir, "free_convection_nde_loss_evolution"))

@info "Animating what the neural network has learned..."
for (id, ds) in coarse_datasets
    filepath = joinpath(output_dir, "learned_free_convection_$id")
    animate_learned_free_convection(ds, NN, free_convection_neural_network, NDEType, T_scaling, wT_scaling, filepath=filepath)
end
