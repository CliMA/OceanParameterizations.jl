using Printf
using Logging
using LinearAlgebra

using ArgParse
using LoggingExtras
using DataDeps
using Flux
using JLD2
using OrdinaryDiffEq

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
            default = "ROCK4"
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

        "--animate-comparisons"
            help = "Produce gif and mp4 animations comparing the different parameterizations for each simulation."
            action = :store_true

        "--animate-nde-solution"
            help = "Produce gif and mp4 animations showing the NDE solution for each simulation."
            action = :store_true

        "--plot-les-flux-fraction"
            help = "Produce plot of the LES diffusive flux fraction."
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

ids_train = args["training-simulations"][1]
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)
validate_simulation_ids(ids_train, ids_test)

animate_comparisons = args["animate-comparisons"]
animate_nde_solution = args["animate-nde-solution"]
plot_les_flux_fraction = args["plot-les-flux-fraction"]

output_dir = args["output-directory"]
mkpath(output_dir)


@info "Planting loggers..."

log_filepath = joinpath(output_dir, "testing.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger


@info "Loading training data..."

data = load_data(ids_train, ids_test, Nz)

datasets = data.datasets
coarse_datasets = data.coarse_datasets


@info "Reading neural network from disk..."

nn_history_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes_history.jld2")
final_nn_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes.jld2")
initial_nn_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes.jld2")

file = jldopen(final_nn_filepath, "r")
NN = file["neural_network"]
T_scaling = file["T_scaling"]
wT_scaling = file["wT_scaling"]
close(file)

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end

file = jldopen(initial_nn_filepath, "r")
initial_NN = file["neural_network"]
close(file)

function free_convection_initial_neural_network(input)
    wT_interior = initial_NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end


@info "Gathering and computing solutions..."

solutions_filepath = joinpath(output_dir, "solutions_and_history.jld2")

if isfile(solutions_filepath)
    @info "Loading solutions from $solutions_filepath..."
    @error "Load the data from the JLD2 file!"
else
    true_solutions = Dict(id => (T=interior(ds["T"])[1, 1, :, :], wT=interior(ds["wT"])[1, 1, :, :]) for (id, ds) in coarse_datasets)
    nde_solutions = Dict(id => solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)
    kpp_solutions = Dict(id => free_convection_kpp(ds) for (id, ds) in coarse_datasets)
    tke_solutions = Dict(id => free_convection_tke_mass_flux(ds) for (id, ds) in coarse_datasets)
    initial_nde_solutions = Dict(id => solve_nde(ds, initial_NN, NDEType, algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)

    convective_adjustment_solutions = Dict(id => oceananigans_convective_adjustment(ds; output_dir) for (id, ds) in coarse_datasets)
    oceananigans_solutions = Dict(id => oceananigans_convective_adjustment_with_neural_network(ds, output_dir=output_dir, nn_filepath=final_nn_filepath) for (id, ds) in coarse_datasets)
end


@info "Plotting loss matrix..."

plot_loss_matrix(coarse_datasets, ids_train, nde_solutions, kpp_solutions, tke_solutions,
                 convective_adjustment_solutions, oceananigans_solutions, T_scaling,
                 filepath_prefix = joinpath(output_dir, "loss_matrix_plots"))

plot_loss_matrix_filled_curves(coarse_datasets, oceananigans_solutions, kpp_solutions, convective_adjustment_solutions, T_scaling,
                               filepath_prefix = joinpath(output_dir, "loss_matrix_filled_curves"))

plot_initial_vs_final_loss_matrix(coarse_datasets, ids_train, nde_solutions, initial_nde_solutions, T_scaling,
                                  filepath_prefix = joinpath(output_dir, "loss_matrix_plots_initial_vs_final"))

plot_initial_vs_final_loss_matrix_filled_curves(coarse_datasets, ids_train, nde_solutions, initial_nde_solutions, T_scaling,
                                                filepath_prefix = joinpath(output_dir, "loss_matrix_plots_online_vs_offline_filled_curves"))

if animate_comparisons
    @info "Animating comparisons..."

    for (id, ds) in coarse_datasets
        filepath = joinpath(output_dir, "free_convection_comparisons_$id")
        plot_comparisons(ds, id, ids_train, nde_solutions[id], kpp_solutions[id], tke_solutions[id],
                         convective_adjustment_solutions[id], oceananigans_solutions[id], T_scaling,
                         filepath = filepath, frameskip = 5)
    end
end


if animate_nde_solution
    @info "Animating NDE solution..."

    for (id, ds) in coarse_datasets
        filepath = joinpath(output_dir, "learned_free_convection_$id")
        animate_learned_free_convection(ds, NN, free_convection_neural_network, NDEType, algorithm, T_scaling, wT_scaling,
                                        filepath=filepath, frameskip=5)
    end
end


@info "Computing NDE solution history..."

if isfile(solutions_filepath)
    @info "Loading NDE solution history from $solutions_filepath..."
    @error "Load the data from the JLD2 file!"
else
    nde_solution_history = compute_nde_solution_history(coarse_datasets, NDEType, algorithm, final_nn_filepath, nn_history_filepath)
end


@info "Plotting loss(epoch)..."

plot_epoch_loss(ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
                filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history"))

plot_epoch_loss_summary(FreeConvection.SIMULATION_IDS, nde_solution_history, true_solutions, T_scaling,
                        filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history_summary"))

plot_epoch_loss_summary_filled_curves(
    FreeConvection.SIMULATION_IDS, nde_solution_history, true_solutions, T_scaling,
    filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history_summary_filled_curves")
)


# @info "Plotting loss(time; epoch)..."

# animate_nde_loss(coarse_datasets, ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
#                  title = "Free convection loss history",
#                  filepath = joinpath(output_dir, "free_convection_nde_loss_evolution"))


if plot_les_flux_fraction
    @info "Comparing advective fluxes ⟨w'T'⟩ with LES diffusive flux ⟨κₑ∂zT⟩..."

    import Plots

    t = coarse_datasets[1]["T"].times ./ 86400
    p = Plots.plot(xlabel="Time (days)", ylabel="|κₑ∂zT| / ( |w'T'| + |κₑ∂zT| )", xlims=extrema(t), grid=false, framestyle=:box,
                   legend=:outertopright, foreground_color_legend=nothing, background_color_legend=nothing, dpi=200)

    for id in FreeConvection.SIMULATION_IDS
        ds = datasets[id]
        advective_heat_flux = sum(ds["wT"].data[1, 1, :, :] .|> abs, dims=1)[:]
        diffusive_heat_flux = sum(ds["κₑ_∂z_T"].data[1, 1, :, :] .|> abs, dims=1)[:]
        total_heat_flux = advective_heat_flux .+ diffusive_heat_flux

        linestyle = 1 <= id <= 9 ? :solid : :dash
        Plots.plot!(p, t, diffusive_heat_flux ./ total_heat_flux, linewidth=2, label="simulation $id", linestyle=linestyle)
    end

    Plots.savefig(joinpath(output_dir, "les_flux_contribution.png"))
end


@info "Saving solutions to JLD2..."

jldopen(solutions_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling

    file["true"] = true_solutions
    file["nde"] = nde_solutions
    file["kpp"] = kpp_solutions
    file["tke"] = tke_solutions
    file["initial_nde"] = initial_nde_solutions
    file["convective_adjustment"] = convective_adjustment_solutions
    file["oceananigans"] = oceananigans_solutions

    file["nde_history"] = nde_solution_history
end
