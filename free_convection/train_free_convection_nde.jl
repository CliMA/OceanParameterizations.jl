using Logging
using Printf
using Random
using Statistics
using LinearAlgebra

using ArgParse
using LoggingExtras
using DataDeps
using Flux
using JLD2
using OrdinaryDiffEq
using Zygote

using Oceananigans
using OceanParameterizations
using FreeConvection

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
ENV["GKSwstype"] = "100"

LinearAlgebra.BLAS.set_num_threads(1)

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--grid-points"
            help = "Number of vertical grid points in the trained neural differential equation (LES data will be coarse-grained to this resolution)."
            default = 32
            arg_type = Int

        "--base-parameterization"
            help = "Base parameterization to use for the NDE. Options: nothing, convective_adjustment"
            default = "nothing"
            arg_type = String

        "--time-stepper"
            help = "DifferentialEquations.jl time stepping algorithm to use."
            default = "Tsit5"
            arg_type = String

        "--output-directory"
            help = "Output directory filepath."
            default = joinpath(@__DIR__, "testing")
            arg_type = String

        "--training-simulations"
            help = "Simulation IDs (list of integers separated by spaces) to train the neural differential equation on." *
                   "All other simulations will be used for testing/validation."
            action = :append_arg
            nargs = '+'
            arg_type = Int
            range_tester = (id -> id in FreeConvection.SIMULATION_IDS)

        "--burn-in-epochs"
            help = "Number of epochs to train on the partial time series."
            default = 0
            arg_type = Int

        "--training-epochs"
            help = "Number of epochs per optimizer to train on the full time series."
            default = 10
            arg_type = Int

        "--conv"
            help = "Toggles filter dim/if a convolutional layer is included in the NN architecture. conv > 1 --> layer is added"
            default = 0
            arg_type = Int

        "--spatial_causality"
            help = "Toggles how/if spatial causality is enforced in dense layer models. Empty string -> not enforced."
            default = ""
            arg_type = String

        "--animate-training-data"
            help = "Produce gif and mp4 animations of each training simulation's data."
            action = :store_true
    end

    return parse_args(settings)
end


@info "Parsing command line arguments..."

args = parse_command_line_arguments()

nde_type = Dict(
    "nothing" => FreeConvectionNDE,
    "convective_adjustment" => ConvectiveAdjustmentNDE
)

Nz = args["grid-points"]
NDEType = nde_type[args["base-parameterization"]]
algorithm = Meta.parse(args["time-stepper"] * "()") |> eval

burn_in_epochs = args["burn-in-epochs"]
full_epochs = args["training-epochs"]

conv = args["conv"]
spatial_causality = args["spatial_causality"]

ids_train = args["training-simulations"][1]
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)
validate_simulation_ids(ids_train, ids_test)

output_dir = args["output-directory"]
mkpath(output_dir)

animate_training_simulations = args["animate-training-data"]

# Save command line arguments used to an executable shell script
open(joinpath(output_dir, "run_three_layer_constant_fluxes.sh"), "w") do io
    write(io, "#!/bin/sh\n")
    write(io, "julia " * basename(@__FILE__) * " " * join(ARGS, " ") * "\n")
end

@info "Planting loggers..."

log_filepath = joinpath(output_dir, "training.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger


@info "Architecting neural network..."

if conv > 1
    NN = Chain(
           x -> reshape(x, Nz, 1, 1, 1),
           Conv((conv, 1), 1 => 1, relu),
           x -> reshape(x, Nz-conv+1),
           Dense(Nz-conv+1, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))
else
    NN = Chain(Dense(Nz, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
end

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end


@info "Loading training data..."

data = load_data(ids_train, ids_test, Nz, convective_adjustment_K=10)

training_datasets = data.training_datasets
coarse_training_datasets = data.coarse_training_datasets

if animate_training_simulations
    @info "Animating ⟨T⟩(z,t) and ⟨w'T⟩(z,t) training data..."

    for id in keys(training_datasets)
        filepath = joinpath(output_dir, "free_convection_training_data_$id")
        if !isfile(filepath * ".mp4") || !isfile(filepath * ".gif")
            animate_training_data(training_datasets[id], coarse_training_datasets[id]; filepath, frameskip=5)
        end
    end
end


@info "Wrangling (T, wT) training data..."

input_training_data = wrangle_input_training_data(coarse_training_datasets)
output_training_data = wrangle_output_training_data(coarse_training_datasets)


@info "Scaling features..."

T_training_data = reduce(hcat, input.temperature for input in input_training_data)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

input_training_data = [rescale(i, T_scaling, wT_scaling) for i in input_training_data]
output_training_data = wT_scaling.(output_training_data)


@info "Batching training data..."

n_training_data = length(input_training_data)
training_data = [(input_training_data[n], output_training_data[:, n]) for n in 1:n_training_data] |> shuffle
data_loader = Flux.Data.DataLoader(training_data, batchsize=n_training_data, shuffle=true)

n_obs = data_loader.nobs
batch_size = data_loader.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Training data loader contains $n_obs pairs of observations (batch size = $batch_size)."


@info "Training neural network on fluxes: ⟨T⟩(z) -> ⟨w′T′⟩(z) mapping..."

causal_penalty = nothing

if spatial_causality == "soft"
    ps = Flux.params(NN)

    dense_layer_idx = 1 + Int(conv > 1) * 3
    dense_layer_params_idx = 1 + Int(conv > 1) * 2

    nrows, ncols = size(ps[dense_layer_params_idx])
    mask = [x < y ? true : false for x in 1:nrows, y in 1:ncols]

    causal_penalty() = sum(abs2, NN[dense_layer_idx].W[mask])

    nn_loss(input, output) = Flux.mse(free_convection_neural_network(input), output) + causal_penalty()
else
    nn_loss(input, output) = Flux.mse(free_convection_neural_network(input), output)
end

nn_training_set_loss(training_data) = mean(nn_loss(input, output) for (input, output) in training_data)

function nn_callback()
    μ_loss = nn_training_set_loss(training_data)
    @info @sprintf("Training free convection neural network... training set MSE loss: μ_loss::%s = %.10e", typeof(μ_loss), μ_loss)
    return μ_loss
end

epochs = 10
optimizers = [ADAM(), Descent(1e-2)]

for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader)
    @info "Training heat flux neural network with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
    Flux.train!(nn_loss, Flux.params(NN), mini_batch, opt, cb=Flux.throttle(nn_callback, 5))
end


@info "Saving initial neural network weights to disk..."

initial_nn_filepath = joinpath(output_dir, "free_convection_initial_neural_network.jld2")

jldopen(initial_nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end


@info "Training neural differential equation on incrementally increasing time spans..."

nn_history_filepath = joinpath(output_dir, "neural_network_history.jld2")

if burn_in_epochs > 0
    training_iterations = (1:20, 1:5:101, 1:10:201, 1:20:401, 1:40:801)
    opt = ADAM()

    for iterations in training_iterations
        @info "Training free convection NDE with iterations=$iterations for $burn_in_epochs epochs  with $(typeof(opt))(η=$(opt.eta))..."
        train_neural_differential_equation!(NN, NDEType, algorithm, coarse_training_datasets, T_scaling, wT_scaling,
                                            iterations, opt, burn_in_epochs, history_filepath=nn_history_filepath, causal_penalty=causal_penalty)
    end
end

@info "Training the neural differential equation on the entire solution while decreasing the learning rate..."

burn_in_iterations = 1:9:1153
optimizers = [ADAM(1e-3)]

for opt in optimizers
    @info "Training free convection NDE with iterations=$burn_in_iterations for $full_epochs epochs with $(typeof(opt))(η=$(opt.eta))..."
    train_neural_differential_equation!(NN, NDEType, algorithm, coarse_training_datasets, T_scaling, wT_scaling,
                                        burn_in_iterations, opt, full_epochs, history_filepath=nn_history_filepath, causal_penalty=causal_penalty)
end


@info "Saving trained neural network weights to disk..."

trained_nn_filepath = joinpath(output_dir, "free_convection_trained_neural_network.jld2")

jldopen(trained_nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end
