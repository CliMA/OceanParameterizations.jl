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

        "--name"
            help = "Experiment name (also determines name of output directory)."
            default = "layers3_depth4_relu_ROCK4"
            arg_type = String

        "--training-simulations"
            help = "Simulation IDs (list of integers separated by spaces) to train the neural differential equation on." *
                   "All other simulations will be used for testing/validation."
            action = :append_arg
            nargs = '+'
            arg_type = Int
            range_tester = (id -> id in FreeConvection.SIMULATION_IDS)
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
experiment_name = args["name"]
NDEType = nde_type[args["base-parameterization"]]
algorithm = Meta.parse(args["time-stepper"] * "()") |> eval

ids_train = args["training-simulations"][1]
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)
validate_simulation_ids(ids_train, ids_test)

output_dir = joinpath(@__DIR__, experiment_name)


@info "Planting loggers..."

log_filepath = joinpath(output_dir, "$(experiment_name)_testing.log")
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


@info "Gathering and computing solutions..."

true_solutions = Dict(id => (T=interior(ds["T"])[1, 1, :, :], wT=interior(ds["wT"])[1, 1, :, :]) for (id, ds) in coarse_datasets)
nde_solutions = Dict(id => solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)
kpp_solutions = Dict(id => free_convection_kpp(ds) for (id, ds) in coarse_datasets)
tke_solutions = Dict(id => free_convection_tke_mass_flux(ds) for (id, ds) in coarse_datasets)

convective_adjustment_solutions = Dict()
oceananigans_solutions = Dict()
for (id, ds) in coarse_datasets
    ca_sol, nn_sol = oceananigans_convective_adjustment_nn(ds, output_dir=output_dir, nn_filepath=final_nn_filepath)
    convective_adjustment_solutions[id] = ca_sol
    oceananigans_solutions[id] = nn_sol
end


@info "Plotting loss matrix..."

plot_loss_matrix(coarse_datasets, ids_train, nde_solutions, kpp_solutions, tke_solutions,
                 convective_adjustment_solutions, oceananigans_solutions, T_scaling,
                 filepath_prefix = joinpath(output_dir, "loss_matrix_plots"))

for (id, ds) in coarse_datasets
    filepath = joinpath(output_dir, "free_convection_comparisons_$id")
    plot_comparisons(ds, id, ids_train, nde_solutions[id], kpp_solutions[id], tke_solutions[id],
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


@info "Plotting loss(epoch)..."

plot_epoch_loss(ids_train, ids_test, nde_solution_history, true_solutions, T_scaling,
                filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history"))

plot_epoch_loss_summary(FreeConvection.SIMULATION_IDS, nde_solution_history, true_solutions, T_scaling,
                        filepath_prefix = joinpath(output_dir, "free_convection_nde_loss_history_summary"))


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
