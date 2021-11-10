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

using FreeConvection: inscribe_history

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

        "--epochs"
            help = "Number of epochs per optimizer to train on the full time series."
            default = 10
            arg_type = Int

        "--animate-training-data"
            help = "Produce gif and mp4 animations of each training simulation's data."
            action = :store_true
    end

    return parse_args(settings)
end


@info "Parsing command line arguments..."

args = parse_command_line_arguments()

base_param = args["base-parameterization"]
use_convective_adjustment = base_param == "convective_adjustment"
use_missing_fluxes = use_convective_adjustment

Nz = args["grid-points"]
epochs = args["epochs"]

ids_train = args["training-simulations"][1]
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)
validate_simulation_ids(ids_train, ids_test)

output_dir = args["output-directory"]
mkpath(output_dir)

animate_training_simulations = args["animate-training-data"]

# Save command line arguments used to an executable shell script
open(joinpath(output_dir, "train_nn_on_fluxes.sh"), "w") do io
    write(io, "#!/bin/sh\n")
    write(io, "julia " * basename(@__FILE__) * " " * join(ARGS, " ") * "\n")
end

@info "Planting loggers..."

log_filepath = joinpath(output_dir, "training_on_fluxes.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger


@info "Architecting neural network..."

NN = Chain(Dense(Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end


@info "Loading training data..."

data = load_data(ids_train, ids_test, Nz)

training_datasets = data.training_datasets
coarse_training_datasets = data.coarse_training_datasets


if use_convective_adjustment
    @info "Computing convective adjustment solutions and fluxes (and missing fluxes)..."

    for (id, ds) in coarse_training_datasets
        sol = oceananigans_convective_adjustment(ds; output_dir)

        grid = ds["T"].grid
        times = ds["T"].times

        # ds.fields["wT"].data .+= ds.fields["κₑ_∂z_T"].data

        ds.fields["T_param"] = FieldTimeSeries(grid, (Center, Center, Center), times, ArrayType=Array{Float32})
        ds.fields["wT_param"] = FieldTimeSeries(grid, (Center, Center, Face), times, ArrayType=Array{Float32})
        ds.fields["wT_missing"] = FieldTimeSeries(grid, (Center, Center, Face), times, ArrayType=Array{Float32})

        ds.fields["T_param"][1, 1, :, :] .= sol.T
        ds.fields["wT_param"][1, 1, :, :] .= sol.wT

        ds.fields["wT_missing"].data .= ds.fields["wT"].data .- ds.fields["wT_param"].data
    end
end


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

input_training_data = wrangle_input_training_data(coarse_training_datasets, use_missing_fluxes=use_missing_fluxes, time_range_skip=1)
output_training_data = wrangle_output_training_data(coarse_training_datasets; use_missing_fluxes, time_range_skip=1)


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

nn_loss(input, output) = Flux.mse(free_convection_neural_network(input), output)

nn_training_set_loss(training_data) = mean(nn_loss(input, output) for (input, output) in training_data)

function nn_callback()
    losses = [nn_loss(input, output) for (input, output) in training_data]

    mean_loss = mean(losses)
    median_loss = median(losses)

    @info @sprintf("Training free convection neural network... training set MSE loss: mean_loss::%s = %.10e, median_loss = %.10e",
                   typeof(mean_loss), mean_loss, median_loss)

    return mean_loss, median_loss
end

optimizers = [ADAM(1e-4)]
history_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes_history.jld2")

for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader)
    @info "Training heat flux neural network with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
    Flux.train!(nn_loss, Flux.params(NN), mini_batch, opt, cb=Flux.throttle(nn_callback, 5))

    mean_loss, median_loss = nn_callback()
    inscribe_history(history_filepath, NN, median_loss)
end


@info "Saving trained neural network weights to disk..."

nn_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes.jld2")

jldopen(nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end
