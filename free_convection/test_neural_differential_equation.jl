using NCDatasets
using Plots
using Oceananigans.Utils
using ClimateParameterizations

function nde_mse(ds, nn_filepath)
    neural_network_parameters = BSON.load(nn_filepath)

    NN = neural_network_parameters[:weights]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    iterations = 1:length(ds["time"])
    true_sol = free_convection_solution(ds, iterations, T_scaling)

    nde = FreeConvectionNDE(NN, ds, 32, iterations)
    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    T₀ = initial_condition(ds, T_scaling)
    sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

    return Flux.mse(sol_npde, true_sol)
end

function plot_nde_mse_in_time(ds, nn_filepath)
    neural_network_parameters = BSON.load(nn_filepath)

    NN = neural_network_parameters[:weights]
    T_scaling = neural_network_parameters[:T_scaling]
    wT_scaling = neural_network_parameters[:wT_scaling]

    iterations = 1:length(ds["time"])
    true_sol = free_convection_solution(ds, iterations, T_scaling)

    nde = FreeConvectionNDE(NN, ds, 32, iterations)
    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    T₀ = initial_condition(ds, T_scaling)
    sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

    error_in_time = Flux.mse(sol_npde, true_sol, agg=x->mean(x, dims=1))

    title = @sprintf("Free convection NDE Q=%dW", Q)
    plot(ds["time"] ./ days, error_in_time[:], linewidth=2, yaxis=:log, ylims=(1e-6, 1e-1),
         label="", title=title, xlabel="Simulation time (days)", ylabel="Mean squared error", dpi=200)
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
    nde_params = FreeConvectionNDEParameters(ds[75], T_scaling, wT_scaling)
    T₀ = initial_condition(ds[75], T_scaling)

    T₀ = initial_condition(ds, T_scaling)
    sol_npde = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array

    time_steps = size(sol_npde, 2)
    anim = @animate for (i, n) in enumerate(iters)
        @info "Plotting $filepath [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(T[:, n], z, linewidth=2, xlim=(19, 20), ylim=(-100, 0),
             label="Oceananigans T(z,t)", xlabel="Temperature (°C)", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(S⁻¹_T.(sol_npde[:, i]), z_coarse, linewidth=2, label="NDE")
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return nothing
end

Nz = 32  # Number of grid points

nn_filepath = "free_convection_neural_differential_equation_trained.bson"

Qs = [25, 50, 75, 100]
Qs_train = [25, 75]
Qs_test = [50, 100]

ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs)

for Q in Qs
    @info @sprintf("Free convection NDE MSE for Q=%dW: %.4e", Q, nde_mse(ds[Q], nn_filepath))
end

plot_nde_mse_in_time(ds[75], nn_filepath)

for Q in Qs
end

# for Q in (Qs_train..., Qs_test...)
#     sol_correct = cat((coarse_grain(ds[Q]["T"][:, n], Nz, Cell) .|> S_T for n in iters_train)..., dims=2)
#     T₀ = coarse_grain(ds[Q]["T"][:, iters_train[1]], Nz, Cell) .|> S_T

#     NN_fast_heat_flux = generate_NN_fast_heat_flux(NN_fast, flux_standarized(0), flux_standarized(Q))
#     npde = construct_neural_pde(NN_fast_heat_flux, ds[Q], standardization, grid_points=Nz, iterations=iters_train)
#     npde.p .= best_weights
#     sol_npde = Array(npde(T₀, npde.p))

#     μ_loss = Flux.mse(sol_npde, sol_correct)
#     @info @sprintf("Q = %dW loss: %e", Q, μ_loss)
# end

animate_learned_free_convection(ds, nn_filepath; grid_points, iterations=nothing, filepath, fps=15)
