using Printf
using Statistics
using BSON
using DifferentialEquations
using Flux
using NCDatasets
using Plots
using Oceananigans.Utils
using OceanParameterizations

function compute_nde_solution_history(ds, Qs, nn_filepath, nn_history_filepath)
    neural_network_parameters = BSON.load(nn_filepath)

    Nz = neural_network_parameters[:grid_points]
    NN = neural_network_parameters[:neural_network]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    nn_history = BSON.load(nn_history_filepath)[:nn_history]

    solution_history = Dict(Q => [] for Q in Qs)

    epochs = length(nn_history)
    nde_params = Dict(Q => FreeConvectionNDEParameters(ds[Q], T_scaling, wT_scaling) for Q in Qs)
    T₀ = Dict(Q => initial_condition(ds[Q], grid_points=Nz, scaling=T_scaling) for Q in Qs)

    for e in 1:epochs
        @info "Computing NDE solutions for epoch $e/$epochs..."
        NN = nn_history[e]
        nde = FreeConvectionNDE(NN, ds[first(Qs)]; grid_points=Nz)
        for Q in Qs
            nde_sol = solve_free_convection_nde(nde, NN, T₀[Q], Tsit5(), nde_params[Q])
            push!(solution_history[Q], nde_sol)
        end
        e % 20 == 0 && GC.gc() # Mercy if you're running this on a little laptop.
    end

    return solution_history
end

function plot_epoch_loss(Qs, nde_solutions, true_solutions)
    epochs = length(nde_solutions[first(Qs)])

    p = plot(dpi=200)
    for Q in Qs
        loss_history = [Flux.mse(true_solutions[Q], nde_solutions[Q][e]) for e in 1:epochs]
        label = @sprintf("Q=%dW", Q)
        title = "Free convection NDE"
        plot!(p, 1:epochs, loss_history, linewidth=2, yaxis=:log, ylims=(1e-3, 10),
              label=label, title=title, xlabel="Epochs", ylabel="Mean squared error",
              legend = :outertopright)
    end

    png_filepath = "free_convection_nde_loss_history.png"
    savefig(png_filepath)

    return p
end

function animate_nde_loss(ds, Qs, nde_solutions, true_solutions, fps=15)
    epochs = length(nde_solutions[first(Qs)])
    times = ds[first(Qs)]["time"] ./ days

    anim = @animate for e in 1:epochs
        @info "Plotting NDE loss evolution... epoch $e/$epochs"
        title = "Training free convection NDE: epoch $e"
        p = plot(dpi=200)
        for Q in Qs
            nde_loss = Flux.mse(true_solutions[Q], nde_solutions[Q][e], agg=x->mean(x, dims=1))[:]
            label = @sprintf("Q=%dW", Q)
            plot!(p, times, nde_loss, linewidth=2, yaxis=:log, ylims=(1e-6, 10),
                  label=label, xlabel="Simulation time (days)", ylabel="Mean squared error",
                  title=title, legend=:bottomright, dpi=200)
        end
    end

    mp4(anim, "free_convection_nde_loss_evolution.mp4", fps=fps)

    return nothing
end

function nde_mse(ds, nn_filepath)
    neural_network_parameters = BSON.load(nn_filepath)

    Nz = neural_network_parameters[:grid_points]
    NN = neural_network_parameters[:neural_network]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    iterations = 1:length(ds["time"])
    true_sol = convection_training_data(ds["T"]; grid_points=Nz, iterations, scaling=T_scaling)

    nde = FreeConvectionNDE(NN, ds, Nz, iterations)
    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    T₀ = initial_condition(ds, T_scaling)
    sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

    return Flux.mse(sol_npde, true_sol)
end

function plot_nde_mse_in_time(ds, nn_filepath)
    neural_network_parameters = BSON.load(nn_filepath)

    Nz = neural_network_parameters[:grid_points]
    NN = neural_network_parameters[:neural_network]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    iterations = 1:length(ds["time"])
    true_sol = convection_training_data(ds["T"]; grid_points=Nz, iterations, scaling=T_scaling)

    nde = FreeConvectionNDE(NN, ds, Nz, iterations)
    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    T₀ = initial_condition(ds, T_scaling)
    sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

    error_in_time = Flux.mse(sol_npde, true_sol, agg=x->mean(x, dims=1))

    title = @sprintf("Free convection NDE Q=%dW", Q)
    p = plot(ds["time"] ./ days, error_in_time[:], linewidth=2, yaxis=:log, ylims=(1e-6, 1e-1),
             label="", title=title, xlabel="Simulation time (days)", ylabel="Mean squared error", dpi=200)

    png_filepath = @sprintf("free_convection_nde_mse_%dW.png", Q)
    savefig(png_filepath)

    return p
end

function plot_nde_mse_in_time(ds, nn_filepath, Qs)
    neural_network_parameters = BSON.load(nn_filepath)

    Nz = neural_network_parameters[:grid_points]
    NN = neural_network_parameters[:neural_network]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    p = plot(dpi=200)

    for Q in Qs
        iterations = 1:length(ds[Q]["time"])
        true_sol = convection_training_data(ds[Q]["T"]; grid_points=Nz, iterations, scaling=T_scaling)

        nde = FreeConvectionNDE(NN, ds[Q], Nz, iterations)
        nde_params = FreeConvectionNDEParameters(ds[Q], T_scaling, wT_scaling)
        T₀ = initial_condition(ds[Q], T_scaling)
        sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

        error_in_time = Flux.mse(sol_npde, true_sol, agg=x->mean(x, dims=1))

        label = @sprintf("Q=%dW", Q)
        title = "Free convection NDE"
        plot!(p, ds[Q]["time"] ./ days, error_in_time[:], linewidth=2, yaxis=:log, ylims=(1e-6, 1),
              label=label, title=title, xlabel="Simulation time (days)", ylabel="Mean squared error",
              legend = :outertopright)
    end

    savefig("free_convection_nde_mse.png")

    return p
end

function animate_learned_free_convection(ds, nn_filepath; grid_points, iterations=nothing, filepath, fps=15)
    neural_network_parameters = BSON.load(nn_filepath)

    NN = neural_network_parameters[:weights]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    T, wT, z = ds["T"], ds["wT"], ds["zC"]
    Nz, Nt = size(T)
    z_coarse = coarse_grain(z, grid_points, Cell)

    if isnothing(iterations)
        iterations = 1:length(ds["time"])
    end

    nde = FreeConvectionNDE(NN, ds, grid_points, iterations)
    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    T₀ = initial_condition(ds, T_scaling)

    T₀ = initial_condition(ds, T_scaling)
    sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

    time_steps = size(sol_npde, 2)
    anim = @animate for (i, n) in enumerate(iterations)
        @info "Plotting $filepath... frame $i/$(length(iterations)) (iteration $n/$Nt)"

        time_str = @sprintf("%.2f days", ds["time"][n] / days)
        title = @sprintf("Free convection NDE Q=%dW: %s", Q, time_str)

        plot(T[:, n], z, linewidth=2, xlim=(19, 20), ylim=(-100, 0),
             label="Oceananigans T(z,t)", xlabel="Temperature (°C)", ylabel="Depth z (meters)",
             title=title, legend=:bottomright, dpi=200)

        T_NN = inv(T_scaling).(sol_npde[:, i])
        plot!(T_NN, z_coarse, linewidth=2, label="NDE")
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return anim
end

nn_filepath = "free_convection_neural_differential_equation_trained.bson"
nn_history_filepath = "free_convection_neural_network_history.bson"

Nz = BSON.load(nn_filepath)[:grid_points]
T_scaling = BSON.load(nn_filepath)[:T_scaling]

Qs = [25, 50, 75, 100]
Qs_train = [25, 75]
Qs_test = [50, 100]

ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs)

true_solutions = Dict(Q => convection_training_data(ds[Q]["T"]; grid_points=32, scaling=T_scaling) for Q in Qs)

solution_history = compute_nde_solution_history(ds, Qs, nn_filepath, nn_history_filepath)
plot_epoch_loss(Qs, solution_history, true_solutions)
animate_nde_loss(ds, Qs, solution_history, true_solutions)

for Q in Qs
    @info @sprintf("Free convection NDE MSE for Q=%dW: %.4e", Q, nde_mse(ds[Q], nn_filepath))
    plot_nde_mse_in_time(ds[Q], nn_filepath)
end

plot_nde_mse_in_time(ds, nn_filepath, Qs)

for Q in Qs
    filepath = @sprintf("free_convection_nde_les_comparison_%dW.mp4", Q)
    animate_learned_free_convection(ds[75], nn_filepath, grid_points=32, filepath=filepath, iterations=1:10:1000)
end
