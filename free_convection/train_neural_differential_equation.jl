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
##### Load weights and feature scaling from train_neural_network.jl
#####

neural_network_parameters = BSON.load("free_convection_neural_network_parameters.bson")

grid_points = neural_network_parameters[:grid_points]
NN = neural_network_parameters[:neural_network]
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
##### Train neural differential equation on incrementally increasing time spans
#####

training_iterations = (1:100, 1:2:201, 1:4:401, 1:8:801)
training_epochs     = (50,    100,     100,     100)

for (iterations, epochs) in zip(training_iterations, training_epochs)

    # Doesn't matter which Q we use to construct the NDE.
    nde = FreeConvectionNDE(NN, ds[first(Qs_train)], grid_points, iterations)

    nde_params = Dict(Q => FreeConvectionNDEParameters(ds[Q], T_scaling, wT_scaling) for Q in Qs_train)
    T₀ = Dict(Q => initial_condition(ds[Q], T_scaling) for Q in Qs_train)
    
    true_sols = Dict(Q => convection_training_data(ds[Q]["T"]; grid_points, iterations, scaling=T_scaling) for Q in Qs_train)
    true_sols = cat([true_sols[Q] for Q in Qs_train]..., dims=2)

    function nde_loss()
        nde_sols = cat([solve_free_convection_nde(nde, NN, T₀[Q], Tsit5(), nde_params[Q]) |> Array for Q in Qs_train]..., dims=2)
        return Flux.mse(nde_sols, true_sols)
    end

    function nde_callback()
        @info @sprintf("Training free convection NDE... loss = %.12e", nde_loss())
        return nothing
    end

    @info "Training free convection NDE with iterations=$iterations for $epochs epochs..."
    Flux.train!(nde_loss, Flux.params(NN), Iterators.repeated((), epochs), ADAM(), cb=nde_callback)
end

# 1:9:length(ds[75]["time"])

#####
##### Save trained neural network to disk
#####

neural_network_parameters = Dict(
       :grid_points => Nz,
    :neural_network => NN,
         :T_scaling => T_scaling,
        :wT_scaling => wT_scaling)

bson("free_convection_neural_differential_equation_trained.bson", neural_network_parameters)
