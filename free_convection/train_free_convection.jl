using Printf
using Random
using Statistics
using LinearAlgebra
using Logging

using NCDatasets
using Plots
using Flux
using BSON
using DifferentialEquations
using DiffEqFlux
using Optim

using ClimateSurrogates
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Utils

using Flux.Data: DataLoader

import DiffEqFlux: FastChain, paramlength, initial_params

using Flux.Data: DataLoader

Logging.global_logger(OceananigansLogger())

ENV["GKSwstype"] = "100"

#####
##### Utils
#####

# Should not have saved constant units as strings...
nc_constant(ds, attr) = parse(Float64, ds.attrib[attr] |> split |> first)

FastLayer(layer) = layer

function FastLayer(layer::Dense)
    N_out, N_in = size(layer.W)
    return FastDense(N_in, N_out, layer.σ, initW=(_,_)->layer.W, initb=_->layer.b)
end

FastChain(NN::Chain) = FastChain([FastLayer(layer) for layer in NN]...)

#####
##### Helper functions
#####

function animate_variable(ds, var, loc; grid_points, xlabel, xlim, filepath, frameskip=1, fps=15)
    if isfile(filepath)
        @info "$filepath exists. Will not animate."
        return nothing
    end
    
    Nz, Nt = size(ds[var])

    if loc == Cell
        z_fine = ds["zC"]
        z_coarse = coarse_grain(ds["zC"], grid_points, Cell)
    elseif loc == Face
        z_fine = ds["zF"]
        z_coarse = coarse_grain(ds["zF"], grid_points+1, Face)
    end

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $var [$n/$Nt]..."
        var_fine = ds[var][:, n]

        if loc == Cell
            var_coarse = coarse_grain(ds[var][:, n], grid_points, Cell)
        elseif loc == Face
            var_coarse = coarse_grain(ds[var][:, n], grid_points+1, Face)
        end

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(var_fine, z_fine, linewidth=2, xlim=xlim, ylim=(-100, 0),
             label="fine (Nz=$(length(z_fine)))", xlabel=xlabel, ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(var_coarse, z_coarse, linewidth=2, label="coarse (Nz=$(length(z_coarse)))")
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return nothing
end

function free_convection_heat_flux_training_data(ds, Qs; grid_points, skip_first=0)
     T = Dict(Q => ds[Q]["T"]  for Q in Qs)
    wT = Dict(Q => ds[Q]["wT"] for Q in Qs)
    Nz, Nt = size(T[Qs[1]])

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    T_inputs, top_flux_inputs, wT_outputs = Dict(), Dict(), Dict()

    for Q in Qs
        ρ₀ = nc_constant(ds[Q], "Reference density")
        cₚ = nc_constant(ds[Q], "Specific_heat_capacity")
        
        top_flux = Q / (ρ₀ * cₚ)
        bot_flux = 0.0

        inds = 1+skip_first:Nt
        top_flux_inputs[Q] = [top_flux for _ in inds]
        T_inputs[Q] = cat((coarse_grain(T[Q][:, n], grid_points, Cell) for n in inds)..., dims=2)
        wT_outputs[Q] = cat((coarse_grain(wT[Q][:, n], grid_points+1, Face) for n in inds)..., dims=2)

        for n in 1:Nt-skip_first
            wT_outputs[Q][1, n] = bot_flux
            wT_outputs[Q][grid_points+1, n] = top_flux
        end
    end

    top_flux_inputs = cat((top_flux_inputs[Q]  for Q in Qs)..., dims=2)
    T_inputs = cat((T_inputs[Q] for Q in Qs)..., dims=2)
    wT_outputs = cat((wT_outputs[Q] for Q in Qs)..., dims=2)

    μ_T, σ_T = mean(T_inputs), std(T_inputs)
    μ_wT, σ_wT = mean(wT_outputs), std(wT_outputs)

    standardize_T(x) = (x - μ_T) / σ_T
    standardize⁻¹_T(y) = σ_T * y + μ_T
    standardize_wT(x) = (x - μ_wT) / σ_wT
    standardize⁻¹_wT(y) = σ_wT * y + μ_wT

    top_flux_inputs = standardize_wT.(top_flux_inputs)
    T_inputs = standardize_T.(T_inputs)
    wT_outputs = standardize_wT.(wT_outputs)

    n_data = length(top_flux_inputs)
    training_data = [((T_inputs[:, n], top_flux_inputs[n]), wT_outputs[:, n]) for n in 1:n_data] |> shuffle

    standardization = (
        T = (μ=μ_T, σ=σ_T, standardize=standardize_T, standardize⁻¹=standardize⁻¹_T),
        wT = (μ=μ_wT, σ=σ_wT, standardize=standardize_wT, standardize⁻¹=standardize⁻¹_wT)
    )

    return training_data, standardization
end

function train_on_heat_flux!(loss, params, training_data, optimizer)
    function cb()
        μ_loss = mean(loss(input, wT) for (input, wT) in training_data)
        @info @sprintf("Training on heat flux... mean(loss) = %e", μ_loss)
        return loss
    end
    
    Flux.train!(loss, params, training_data, optimizer, cb=Flux.throttle(cb, 2))

    return nothing
end

function animate_learned_heat_flux(ds, NN, standardization; grid_points, filepath, frameskip=1, fps=15)
    T, wT, zF = ds["T"], ds["wT"], ds["zF"]
    Nz, Nt = size(T)
    zF_coarse = coarse_grain(zF, grid_points+1, Face)

    Q  = nc_constant(ds, "Heat flux")
    ρ₀ = nc_constant(ds, "Reference density")
    cₚ = nc_constant(ds, "Specific_heat_capacity")
    top_flux = Q / (ρ₀ * cₚ)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting learned heat flux [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(wT[:, n], zF, linewidth=2, xlim=(-1e-5, 3e-5), ylim=(-100, 0),
             label="Oceananigans wT", xlabel="Heat flux", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        S_T = standardization.T.standardize
        S_wT, S⁻¹_wT = standardization.wT.standardize, standardization.wT.standardize⁻¹

        if NN isa DiffEqFlux.FastChain
            T_NN = coarse_grain(T[:, n], grid_points, Cell) .|> S_T
            wT_NN = NN(T_NN, npde.p) .|> S⁻¹_wT
            plot!(wT_NN, zF_coarse, linewidth=2, label="Neural network")
        else
            input = (coarse_grain(T[:, n], grid_points, Cell) .|> S_T, top_flux .|> S_wT)
            wT_NN = input |> NN .|> S⁻¹_wT
            plot!(wT_NN, zF_coarse, linewidth=2, label="Neural network")
        end
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return nothing
end

""" Returns a discrete 1D derivative operator for cell center to cell (f)aces. """
function Dᶠ(N, Δ)
    Dᶠ = zeros(N, N+1)
    for k in 1:N
        Dᶠ[k, k]   = -1.0
        Dᶠ[k, k+1] =  1.0
    end
    Dᶠ = 1/Δ * Dᶠ
    return Dᶠ
end

""" Returns a discrete 1D derivative operator for cell faces to cell (c)enters. """
function Dᶜ(N, Δ)
    Dzᶜ = zeros(N+1, N)
    for k in 2:N
        Dᶜ[k, k-1] = -1.0
        Dᶜ[k, k]   =  1.0
    end
    Dᶜ = 1/Δ * Dᶜ
    return Dᶜ
end

function construct_neural_pde(NN, ds, standardization; grid_points, time_steps)
    H = abs(ds["zF"][1])
    τ = ds["time"][end]

    Nz = grid_points
    zC = coarse_grain(ds["zC"], Nz, Cell)
    Δẑ = diff(zC)[1] / H

    Dzᶠ = Dᶠ(Nz, Δẑ)

    # Set up neural network for non-dimensional PDE
    # ∂T/dt = - ∂z(wT) + ...
    σ_T, σ_wT = standardization.T.σ, standardization.wT.σ

    NN_∂T∂t = FastChain(NN,
                        (wT, _) -> - Dzᶠ * wT,
                        (∂z_wT, _) -> σ_wT/σ_T * τ/H * ∂z_wT)

    # Set up and return neural differential equation
    Nt = length(ds["time"])
    tspan = (0.0, maximum(time_steps) / Nt)
    tsteps = range(tspan[1], tspan[2], length = length(time_steps))

    return NeuralODE(NN_∂T∂t, tspan, ROCK4(), reltol=1e-3, saveat=tsteps)
end

function train_free_convection_neural_pde!(npde, training_data, optimizers; epochs=1)
    time_steps = length(npde.kwargs[:saveat])
    sol_correct = cat([training_data[n][1] for n in 1:time_steps]..., dims=2)

    T₀ = training_data[1][1]

    H  = abs(ds["zF"][1])
    Nz = length(T₀)
    zF = coarse_grain(ds["zF"], Nz, Face)
    Δẑ = diff(zF)[2] / H

    Dzᶜ = Dᶜ(Nz, Δẑ)

    function loss(θ)
        sol_npde = Array(npde(T₀, θ))
        Nt = size(sol_npde)[2]
        dTdz = cat([Dzᶜ * sol_npde[:, n] for n in 1:Nt]..., dims=2)

        C = 0.5  # loss2 will always be weighted with 0 <= weight <= C
        loss1 = Flux.mse(sol_npde, sol_correct)
        loss2 = mean(min.(dTdz, 0) .^ 2)
        weighted_loss = loss1 * (1 + min(loss2, C*loss1))

        # println("loss1=$loss1, loss2=$loss2, weighted_loss=$weighted_loss")
        return weighted_loss
    end

    function cb(θ, args...)
        @info @sprintf("Training free convection neural PDE... loss = %e", loss(θ))
        return false
    end

    # Train!
    for opt in optimizers
        if opt isa Optim.AbstractOptimizer
            for e in 1:epochs
                @info "Training free convection neural PDE for $(time_steps-1) time steps with $(typeof(opt)) [epoch $e/$epochs]..."
                res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=Flux.throttle(cb, 2))
                display(res)
                npde.p .= res.minimizer
            end
        else
            for e in 1:epochs
                @info "Training free convection neural PDE for $(time_steps-1) time steps with $(typeof(opt))(η=$(opt.eta)) [epoch $e/$epochs]..."
                res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=Flux.throttle(cb, 2), maxiters=100)
                display(res)
                npde.p .= res.minimizer
            end
        end
    end
end

function animate_learned_free_convection(ds, npde, standardization; grid_points, skip_first, frameskip=1, fps=15)
    T, wT, z = ds["T"], ds["wT"], ds["zC"]
    Nz, Nt = size(T)
    z_coarse = coarse_grain(z, grid_points, Cell)

    S_T, S⁻¹_T = standardization.T.standardize, standardization.T.standardize⁻¹

    T₀_NN = coarse_grain(T[:, 1], grid_points, Cell) .|> S_T
    sol_npde = npde(T₀_NN) |> Array

    time_steps = size(sol_npde, 2)
    anim = @animate for n=1:frameskip:time_steps
        @info "Plotting learned free convection [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n+skip_first] / day)

        plot(T[:, n+skip_first], z, linewidth=2, xlim=(19, 20), ylim=(-100, 0),
             label="Oceananigans T(z,t)", xlabel="Temperature (°C)", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(S⁻¹_T.(sol_npde[:, n]), z_coarse, linewidth=2, label="Neural PDE")
    end

    filename = "free_convection_neural_pde.mp4"
    @info "Saving $filename"
    mp4(anim, filename, fps=fps)

    return nothing
end

#####
##### Problem parameters
#####

Nz = 32  # Number of grid points for the neural PDE.

skip_first = 5  # Number of transient time snapshots to skip at the start 
future_time_steps = 1

#####
##### Pick training and testing data
#####

Qs = (25, 50, 75, 100)
Qs_train = (25, 75)
Qs_test  = (50,)

#####
##### Load and animate data
#####

ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs)

for Q in Qs
    T_filepath = "free_convection_T_$(Q)W.mp4"
    animate_variable(ds[Q], "T", Cell, grid_points=Nz, xlabel="Temperature T (°C)", xlim=(19, 20),
                     filepath=T_filepath, frameskip=5)

    wT_filepath = "free_convection_wT_$(Q)W.mp4"
    animate_variable(ds[Q], "wT", Face, grid_points=Nz, xlabel="Heat flux wT (m/s °C)", xlim=(-1e-5, 3e-5),
                     filepath=wT_filepath, frameskip=5)
end

#####
##### Prepare heat flux training data
#####

training_data_heat_flux, standardization =
    free_convection_heat_flux_training_data(ds, Qs_train, grid_points=Nz, skip_first=skip_first)

n_obs = length(training_data_heat_flux)
@info "Heat flux training data contains $n_obs pairs."

data_loader_heat_flux = Flux.Data.DataLoader(training_data_heat_flux, batchsize=n_obs, shuffle=true)

n_obs = data_loader_heat_flux.nobs
batch_size = data_loader_heat_flux.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Heat flux data loader contains $n_obs observations (batch size = $batch_size)."

#####
##### Define temperature T -> heat flux wT neural network
#####

NN_heat_flux_filepath = "NN_heat_flux.bson"

if isfile(NN_heat_flux_filepath)
    @info "Loading $NN_heat_flux_filepath..."
    BSON.@load NN_heat_flux_filepath NN
else
    NN = Chain(Dense( Nz, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
end

bot_flux = standardization.wT.standardize(0)

function NN_heat_flux(input)
    T, top_flux = input
    ϕ = NN(T)
    wT = cat(bot_flux, ϕ, top_flux, dims=1)
    return wT
end

#####
##### Train neural network on temperature T -> heat flux wT mapping
#####

if !isfile(NN_heat_flux_filepath)
    loss_heat_flux(input, wT) = Flux.mse(NN_heat_flux(input), wT)

    epochs = 1
    optimizers = [ADAM(1e-2)]

    for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader_heat_flux)
        @info "Training heat flux with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
        train_on_heat_flux!(loss_heat_flux, Flux.params(NN), mini_batch, opt)
    end

    @info "Saving $NN_heat_flux_filepath..."
    BSON.@save NN_heat_flux_filepath NN

    for Q in Qs_train
        filepath = "learned_heat_flux_Q$(Q)W.mp4"
        animate_learned_heat_flux(ds[Q], NN_heat_flux, standardization, grid_points=Nz, filepath=filepath, frameskip=5, fps=15)
    end
end

#####
##### Prepare free convection T(z, t) training data
#####

# training_data_time_step =
#     free_convection_time_step_training_data(ds, standardization, grid_points=Nz, future_time_steps=future_time_steps)
# 
# @info "Time step training data contains $(length(training_data_heat_flux)) time steps."

NN_fast = FastChain(NN)

function NN_fast_heat_flux(input, p)
    T, top_flux = input
    ϕ = NN_fast(T, p)
    wT = cat(bot_flux, ϕ, top_flux, dims=1)
    return wT
end

DiffEqFlux.paramlength(::typeof(NN_fast_heat_flux)) = paramlength(NN_fast)
DiffEqFlux.initial_params(::typeof(NN_fast_heat_flux)) = initial_params(NN_fast)

npde = construct_neural_pde(NN_fast_heat_flux, ds[25], standardization, grid_points=Nz, time_steps=0:50)

error("Stop yo")

for (Nt, epochs) in zip((50, 100, 200, 350, 500, 750, 1000), (2, 1, 1, 1, 1, 1, 1))
    global npde
    new_npde = construct_neural_pde(NN_fast, ds, standardization, grid_points=Nz, time_steps=Nt)
    new_npde.p .= npde.p; npde = new_npde; # Keep using the same weights/parameters!
    train_free_convection_neural_pde!(npde, training_data_time_step, [ADAM(1e-3)], epochs=epochs)

    animate_learned_free_convection(ds, npde, standardization, grid_points=Nz, skip_first=skip_first)
    animate_learned_heat_flux(ds, FastChain(npde.model.layers[1:end-2]...), standardization, grid_points=Nz, frameskip=5, fps=15)
end

npde_filename = "free_convection_neural_pde_parameters.bson"
@info "Saving $npde_filename..."
npde_params = npde.p
BSON.@save npde_filename npde_params
