using Printf
using Logging

using ArgParse
using LoggingExtras
using DataDeps
using JLD2
using ProgressMeter
using OrdinaryDiffEq
using CairoMakie

using Oceananigans
using OceanParameterizations
using FreeConvection

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
output_dir = joinpath(@__DIR__, experiment_name)

@info "Planting loggers..."

log_filepath = joinpath(output_dir, "$(experiment_name)_testing.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger


@info "Loading training data..."

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
coarse_datasets = data.coarse_datasets

function animate_training_data(dss; filepath, frameskip=1, fps=30, legend)

    times = dss[1]["T"].times

    frame = Node(1)
    fig = Figure(resolution=(1920, 1080))

    ax1 = fig[1, 1] = Axis(fig, title="Heat flux", xlabel="T (°C)", ylabel="z (m)")
    ax2 = fig[1, 2] = Axis(fig, title="Temperature", xlabel="w'T' (m/s K)")

    for (id, ds) in dss
        zc = znodes(ds["T"])[:]
        zf = znodes(ds["wT"])[:]

        T = @lift interior(ds["T"])[1, 1, :, $frame]
        wT = @lift interior(ds["wT"])[1, 1, :, $frame]

        lines!(ax1, wT, zf)
        lines!(ax2, T, zc)
    end

    title = @lift "Free convection training data: t = $(prettytime(times[$frame]))"
    supertitle = fig[0, :] = Label(fig, title, textsize=30)

    frames = 1:frameskip:length(times)
    record(fig, filepath, frames, framerate=fps) do n
        @info "Animating free convection training data frame $n/$(length(times))..."
        frame[] = n
    end

    return nothing
end
