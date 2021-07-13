using Printf
using Logging

using ArgParse
using LoggingExtras
using DataDeps
using Flux
using JLD2
using OrdinaryDiffEq

using Oceananigans
using OceanParameterizations
using FreeConvection

ENV["GKSwstype"] = "100"

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
coarse_datasets = data.coarse_datasets


@info "Reading neural network from disk..."

nn_history_filepath = joinpath(output_dir, "neural_network_history.jld2")
final_nn_filepath = joinpath(output_dir, "free_convection_final_neural_network.jld2")
initial_nn_filepath = joinpath(output_dir, "free_convection_initial_neural_network.jld2")

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

true_solutions = Dict(id => (T=interior(ds["T"])[1, 1, :, :], wT=interior(ds["wT"])[1, 1, :, :]) for (id, ds) in coarse_datasets)
nde_solutions = Dict(id => solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)
kpp_solutions = Dict(id => free_convection_kpp(ds) for (id, ds) in coarse_datasets)
tke_solutions = Dict(id => free_convection_tke_mass_flux(ds) for (id, ds) in coarse_datasets)
initial_nde_solutions = Dict(id => solve_nde(ds, initial_NN, NDEType, algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)

convective_adjustment_solutions = Dict(id => oceananigans_convective_adjustment(ds; output_dir) for (id, ds) in coarse_datasets)
oceananigans_solutions = Dict(id => oceananigans_convective_adjustment_with_neural_network(ds, output_dir=output_dir, nn_filepath=final_nn_filepath) for (id, ds) in coarse_datasets)


@info "Plotting loss matrix..."

plot_loss_matrix(coarse_datasets, ids_train, nde_solutions, kpp_solutions, tke_solutions,
                 convective_adjustment_solutions, oceananigans_solutions, T_scaling,
                 filepath_prefix = joinpath(output_dir, "loss_matrix_plots.png"))

plot_initial_vs_final_loss_matrix(coarse_datasets, ids_train, nde_solutions, initial_nde_solutions, T_scaling,
                                  filepath_prefix = joinpath(output_dir, "loss_matrix_plots_initial_vs_final.png"))


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

nde_solution_history = compute_nde_solution_history(coarse_datasets, NDEType, algorithm, final_nn_filepath, nn_history_filepath)

nde_history_filepath = joinpath(output_dir, "free_convection_nde_history.jld2")

jldopen(nde_history_filepath, "w") do file
    file["nde_solution_history"] = nde_solution_history
end

close(file)


@info "Plotting loss(epoch)..."

plot_epoch_loss(ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
                filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history"))

plot_epoch_loss_summary(FreeConvection.SIMULATION_IDS, nde_solution_history, true_solutions, T_scaling,
                        filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history_summary"))

plot_epoch_loss_summary_filled_curves(
    FreeConvection.SIMULATION_IDS, nde_solution_history, true_solutions, T_scaling,
    filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history_summary_filled_curves")
)


@info "Plotting loss(time; epoch)..."

animate_nde_loss(coarse_datasets, ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
                 title = "Free convection loss history",
                 filepath = joinpath(output_dir, "free_convection_nde_loss_evolution"))

#=
@info "Comparing advective fluxes ⟨w'T'⟩ with LES diffusive flux ⟨κₑ∂zT⟩..."

# t = coarse_datasets[1]["T"].times ./ 86400
# p = plot(xlabel="Time (days)", ylabel="|κₑ∂zT| / ( |w'T'| + |κₑ∂zT| )", xlims=extrema(t), grid=false, framestyle=:box,
#          legend=:outertopright, foreground_color_legend=nothing, background_color_legend=nothing, dpi=200)

# for (id, ds) in coarse_datasets
#     advective_heat_flux = sum(ds[:wT].data .|> abs, dims=1)[:]
#     diffusive_heat_flux = sum(ds[:κₑ_∂z_T].data .|> abs, dims=1)[:]
#     total_heat_flux = advective_heat_flux .+ diffusive_heat_flux
#     label = @sprintf("%d W/m²", -ds.metadata[:heat_flux_Wm⁻²])
#     plot!(p, t, diffusive_heat_flux ./ total_heat_flux, linewidth=2, label=label)
# end

# savefig(joinpath(output_dir, "les_flux_contribution.png"))
=#

# TODO: Save solutions to disk.
