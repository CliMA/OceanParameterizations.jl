using Printf
using Logging

using ArgParse
using LoggingExtras
using DataDeps
using GeoData
using Flux
using JLD2
using OrdinaryDiffEq

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
            arg_type = String

        "--time-stepper"
            help = "DifferentialEquations.jl time stepping algorithm to use."
            default = "ROCK4"
            arg_type = String

        "--name"
            help = "Experiment name (also determines name of output directory)."
            default = "layers3_depth4_relu_ROCK4"
            arg_type = String
    end

    return parse_args(settings)
end

## Parse command line arguments

@info "Parsing command line arguments..."
args = parse_command_line_arguments()

Nz = args["grid-points"]
experiment_name = args["name"]

output_dir = joinpath(@__DIR__, experiment_name)
mkpath(output_dir)

log_filepath = joinpath(output_dir, "$(experiment_name)_testing.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger

nde_type = Dict(
    "free_convection" => FreeConvectionNDE,
    "convective_adjustment" => ConvectiveAdjustmentNDE
)

NDEType = nde_type[args["nde"]]
algorithm = Meta.parse(args["time-stepper"] * "()") |> eval

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

## Filepaths

@info "Reading neural network..."

nn_history_filepath = joinpath(output_dir, "neural_network_history.jld2")
final_nn_filepath = joinpath(output_dir, "free_convection_final_neural_network.jld2")

final_nn = jldopen(final_nn_filepath, "r")
NN = final_nn["neural_network"]
T_scaling = final_nn["T_scaling"]
wT_scaling = final_nn["wT_scaling"]
close(final_nn)

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end

## Gather solutions

@info "Gathering and computing solutions..."

nde_solutions = Dict(id => solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)
true_solutions = Dict(id => (T=ds[:T].data, wT=ds[:wT].data) for (id, ds) in coarse_datasets)
kpp_solutions = Dict(id => free_convection_kpp(ds) for (id, ds) in coarse_datasets)
tke_solutions = Dict(id => free_convection_tke_mass_flux(ds) for (id, ds) in coarse_datasets)

convective_adjustment_solutions = Dict()
oceananigans_solutions = Dict()
for (id, ds) in coarse_datasets
    ca_sol, nn_sol = oceananigans_convective_adjustment_nn(ds, output_dir=output_dir, nn_filepath=final_nn_filepath)
    convective_adjustment_solutions[id] = ca_sol
    oceananigans_solutions[id] = nn_sol
end

plot_loss_matrix(coarse_datasets, ids_train, nde_solutions, kpp_solutions, tke_solutions,
                 convective_adjustment_solutions, oceananigans_solutions, T_scaling,
                 filepath = joinpath(output_dir, "loss_matrix_plots.png"))

for (id, ds) in coarse_datasets
    filepath = joinpath(output_dir, "free_convection_comparisons_$id")
    plot_comparisons(ds, nde_solutions[id], kpp_solutions[id], tke_solutions[id],
                     convective_adjustment_solutions[id], oceananigans_solutions[id], T_scaling,
                     filepath = filepath, frameskip = 5)
end

@info "Animating what the neural network has learned..."
for (id, ds) in coarse_datasets
    filepath = joinpath(output_dir, "learned_free_convection_$id")
    animate_learned_free_convection(ds, NN, free_convection_neural_network, NDEType, algorithm, T_scaling, wT_scaling,
                                    filepath=filepath, frameskip=5)
end

@info "Computing NDE solution history..."

nde_solution_history = compute_nde_solution_history(coarse_datasets, NDEType, algorithm, final_nn_filepath, nn_history_filepath)

plot_epoch_loss(ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
                title = "Free convection loss history",
                filepath = joinpath(output_dir, "free_convection_nde_loss_history.png"))

animate_nde_loss(coarse_datasets, ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
                 title = "Free convection loss history",
                 filepath = joinpath(output_dir, "free_convection_nde_loss_evolution"))
