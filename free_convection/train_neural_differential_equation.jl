using Printf
using Statistics

using BSON
using Flux
using DifferentialEquations
using DiffEqFlux
using NCDatasets
using ClimateParameterizations

using Oceananigans.Grids: Cell, Face

#####
##### Neural differential equation parameters
#####

Nz = 32  # Number of grid points

#####
##### Load weights and feature scaling from train_neural_network.jl
#####

neural_network_parameters = BSON.load("free_convection_neural_network_parameters.bson")

NN = neural_network_parameters[:weights]
T_scaling = neural_network_parameters[:T_scaling]
wT_scaling = neural_network_parameters[:wT_scaling]

#####
##### Load training data
#####

# Choose which free convection simulations to train on.
Qs_train = [25, 75]

# Load NetCDF data for each simulation.
ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs_train)

#####
##### Utils for training the neural differential equation
#####

nde_training_data(ds, iterations, T_scaling) =
    cat([T_scaling.(coarse_grain(ds["T"][:, i], Nz, Cell)) for i in iterations]..., dims=2)

function train_free_convection_nde!(NN, nde, nde_params, T₀, true_sol, epochs, opt)
    function nde_loss()
        nde_sol = solve_free_convection_nde(nde, NN, T₀, Tsit5(), nde_params) |> Array
        return Flux.mse(nde_sol, true_sol)
    end

    function nde_callback()
        @info @sprintf("Training free convection NDE... loss = %.12e", nde_loss())
        return nothing
    end

    nn_params = Flux.params(NN)
    Flux.train!(nde_loss, nn_params, Iterators.repeated((), epochs), opt, cb=nde_callback)
end

#####
##### Train neural differential equation
#####

training_iterations = (1:50, 1:100, 1:2:201, 1:4:401, 1:8:801)
training_epochs     = (50,   50,    100,     100,     100)

for (iterations, epochs) in zip(training_iterations, training_epochs)
    nde = FreeConvectionNDE(NN, ds[75], Nz, iterations)
    nde_params = FreeConvectionNDEParameters(ds[75], T_scaling, wT_scaling)
    T₀ = initial_condition(ds[75], T_scaling)
    true_sol = nde_training_data(ds, iterations, T_scaling)

    @info "Training free convection NDE with iterations=$iterations for $epochs epochs..."
    train_free_convection_nde!(NN, nde, nde_params, T₀, true_sol, epochs, ADAM())
end

# 1:9:length(ds[75]["time"])

#####
##### Save trained neural network to disk
#####

neural_network_parameters =
    Dict(   :weights => NN,
          :T_scaling => T_scaling,
         :wT_scaling => wT_scaling)

bson("free_convection_neural_differential_equation_trained.bson", neural_network_parameters)

#=

#####
##### Train on multiple simulations at once while incrementally increasing the time span
#####

training_intervals = (1:50, 1:100, 1:2:201, 1:4:401, 1:8:801, 1:9:length(ds[25]["time"]))
training_maxiters  = (50,   50,    100,     100,     100,     100)
training_epochs    = (1,    2,     2,       2,       2,       3)

training_intervals = [1:9:length(ds[25]["time"])]
training_maxiters  = [100]
training_epochs    = [10]

training_intervals = [1:9:length(ds[25]["time"])]
training_maxiters  = [500]
training_epochs    = [1]

for (iters_train, maxiters, epochs) in zip(training_intervals, training_maxiters, training_epochs), e in 1:epochs
    global best_weights

    training_data_time_step = cat([cat((coarse_grain(ds[Q]["T"][:, n], Nz, Cell) .|> S_T for n in iters_train)..., dims=2) for Q in Qs_train]..., dims=2)

    T₀s = Dict(Q => coarse_grain(ds[Q]["T"][:, iters_train[1]], Nz, Cell) .|> S_T for Q in Qs_train)

    NNs_fast_heat_flux = Dict(
        Q => generate_NN_fast_heat_flux(NN_fast, flux_standarized(0), flux_standarized(Q))
        for Q in Qs_train
    )

    npdes = Dict(
        Q => construct_neural_pde(NNs_fast_heat_flux[Q], ds[Q], standardization, grid_points=Nz, iterations=iters_train)
        for Q in Qs_train
    )

    for Q in Qs_train
        npdes[Q].p .= best_weights
    end

    function combined_loss(θ)
        sols_npde = cat([Array(npdes[Q](T₀s[Q], θ)) for Q in Qs_train]..., dims=2)
        dTdz = cat([Dzᶜ * sols_npde[:, n] for n in 1:size(sols_npde, 2)]..., dims=2)

        C = 5  # loss_dTdz will always be weighted with 0 <= weight <= C
        loss_T = Flux.mse(sols_npde, training_data_time_step)
        loss_dTdz = mean(min.(dTdz, 0) .^ 2)
        weighted_loss = loss_T + min(C * loss_T, loss_dTdz)

        return weighted_loss, loss_T, loss_dTdz
    end

    @info "Training free convection neural PDE for iterations $iters_train (epoch $e/$epochs)..."
    η = (epochs - e + 1) * 1e-3
    train_free_convection_neural_pde!(npdes[Qs_train[1]], combined_loss, ADAM(η), maxiters=maxiters)

    best_weights .= npdes[Qs_train[1]].p
end

npde_filename = "free_convection_neural_pde_parameters.bson"
@info "Saving $npde_filename..."
BSON.@save npde_filename best_weights

#####
##### Quantify testing and training errors
#####

for Q in (Qs_train..., Qs_test...)
    iters_train = training_intervals[end]
    sol_correct = cat((coarse_grain(ds[Q]["T"][:, n], Nz, Cell) .|> S_T for n in iters_train)..., dims=2)
    T₀ = coarse_grain(ds[Q]["T"][:, iters_train[1]], Nz, Cell) .|> S_T

    NN_fast_heat_flux = generate_NN_fast_heat_flux(NN_fast, flux_standarized(0), flux_standarized(Q))
    npde = construct_neural_pde(NN_fast_heat_flux, ds[Q], standardization, grid_points=Nz, iterations=iters_train)
    npde.p .= best_weights
    sol_npde = Array(npde(T₀, npde.p))

    μ_loss = Flux.mse(sol_npde, sol_correct)
    @info @sprintf("Q = %dW loss: %e", Q, μ_loss)
end

#####
##### Animate learned heat flux and free convection solutions on training and testing simulations
#####

for Q in Qs
    regime = Q in Qs_train ? "training" : "testing"

    iters_train = training_intervals[end]

    bot_flux_S = flux_standarized(0)
    top_flux_S = flux_standarized(Q)

    NN_fast_heat_flux = generate_NN_fast_heat_flux(NN_fast, bot_flux_S, top_flux_S)
    npde = construct_neural_pde(NN_fast_heat_flux, ds[Q], standardization, grid_points=Nz, iterations=iters_train)
    npde.p .= best_weights

    filepath = "free_convection_neural_pde_$(regime)_$(Q)W.mp4"
    animate_learned_free_convection(ds[Q], npde, standardization, grid_points=Nz, iters=iters_train, filepath=filepath)

    filepath = "learned_heat_flux_$(regime)_$(Q)W.mp4"
    animate_learned_heat_flux(ds[Q], FastChain(npde.model.layers[1:end-2]...), standardization, grid_points=Nz, filepath=filepath, frameskip=5, fps=15, npde=npde)
end
