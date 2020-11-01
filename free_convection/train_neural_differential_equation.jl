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
