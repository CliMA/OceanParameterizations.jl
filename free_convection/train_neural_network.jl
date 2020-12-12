using Printf
using Random
using Statistics

using BSON
using Flux
using NCDatasets
using OceanParameterizations
using FreeConvection

#####
##### Neural differential equation parameters
#####

Nz = 32  # Number of grid points

#####
##### Neural network architecture T (Nz grid points) -> wT (Nz+1 grid points)
#####

NN = Chain(Dense( Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

NN_params = Flux.params(NN)

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end

#####
##### Load training data
#####

# Choose which free convection simulations to train on.
Qs_train = [25, 75]

# Load NetCDF data for each simulation.
ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs_train)

# Load input training data (temperature T + bottom and top heat fluxes)
input_training_data = []

for Q in Qs_train
    ρ₀ = nc_constant(ds[Q].attrib["Reference density"])
    cₚ = nc_constant(ds[Q].attrib["Specific_heat_capacity"])

    T_Q = convection_training_data(ds[Q]["T"], grid_points=Nz)
    bottom_flux = 0.0
    top_flux = Q / (ρ₀ * cₚ)

    input_training_data_Q = [FreeConvectionTrainingDataInput(T_Q[:, n], bottom_flux, top_flux)
                             for n in 1:size(T_Q, 2)]

    push!(input_training_data, input_training_data_Q)
end

input_training_data = cat(input_training_data..., dims=1)

# Load output training data (heat flux wT).
output_training_data = []

for Q in Qs_train
    ρ₀ = nc_constant(ds[Q].attrib["Reference density"])
    cₚ = nc_constant(ds[Q].attrib["Specific_heat_capacity"])

    output_training_data_Q = convection_training_data(ds[Q]["wT"], grid_points=Nz+1)

    # We need to add the imposed surface heat flux.
    top_flux = Q / (ρ₀ * cₚ)
    output_training_data_Q[end, :] .= top_flux

    push!(output_training_data, output_training_data_Q)
end

output_training_data = cat(output_training_data..., dims=2)

#####
##### Feature scaling
#####

T_training_data = cat([input.temperature for input in input_training_data]..., dims=2)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

input_training_data = [rescale(i, T_scaling, wT_scaling) for i in input_training_data]
output_training_data = wT_scaling.(output_training_data)

#####
##### Construct training data pairs
#####

n_training_data = length(input_training_data)
training_data = [(input_training_data[n], output_training_data[:, n]) for n in 1:n_training_data] |> shuffle
data_loader = Flux.Data.DataLoader(training_data, batchsize=n_training_data, shuffle=true)

n_obs = data_loader.nobs
batch_size = data_loader.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Training data loader contains $n_obs pairs of observations (batch size = $batch_size)."

#####
##### Train neural network on T -> wT mapping
#####

nn_loss(input, output) = Flux.mse(free_convection_neural_network(input), output)

nn_training_set_loss(training_data) = mean(nn_loss(input, output) for (input, output) in training_data)

function nn_callback()
    μ_loss = nn_training_set_loss(training_data)
    @info @sprintf("Training free convection neural network... mean training set MSE loss = %.12e", μ_loss)
    return μ_loss
end

epochs = 2
optimizers = [ADAM(1e-3), Descent(1e-4)]

for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader)
    @info "Training heat flux with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
    Flux.train!(nn_loss, NN_params, mini_batch, opt, cb=Flux.throttle(nn_callback, 5))
end

#####
##### Animate the heat flux the neural network has learned
#####

for Q in Qs_train
    filepath = "learned_heat_flux_initial_guess_Q$(Q)W.mp4"
    animate_learned_heat_flux(ds[Q], free_convection_neural_network, T_scaling, wT_scaling, grid_points=Nz, filepath=filepath, fps=15)
end

#####
##### Save neural network + weights
#####

neural_network_parameters = Dict(
       :grid_points => Nz,
    :neural_network => NN,
         :T_scaling => T_scaling,
        :wT_scaling => wT_scaling)

bson("free_convection_neural_network_parameters.bson", neural_network_parameters)
