using Printf
using Random
using Statistics
using LinearAlgebra

using NCDatasets
using Plots
using Flux
using BSON
using DifferentialEquations
using DiffEqFlux
using Optim

using ClimateSurrogates
using Oceananigans.Utils

import DiffEqFlux: FastChain

using Flux.Data: DataLoader

ENV["GKSwstype"] = "100"

include("free_convection_npde.jl")

function animate_variable(ds, var; grid_points, xlabel, xlim, frameskip, fps)
    Nz, Nt = size(ds[var])

    z_fine = ds["zC"]
    z_coarse = coarse_grain(ds["zC"], grid_points)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $var [$n/$Nt]..."
        var_fine = ds[var][:, n]
        var_coarse = coarse_grain(ds[var][:, n], grid_points)

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(var_fine, z_fine, linewidth=2, xlim=xlim, ylim=(-100, 0),
             label="fine (Nz=$Nz)", xlabel=xlabel, ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(var_coarse, z_coarse, linewidth=2, label="coarse (Nz=$grid_points)")
    end

    filename = "free_convection_$var.mp4"
    @info "Saving $filename"
    mp4(anim, filename, fps=fps)

    return nothing
end

function free_convection_heat_flux_training_data(ds; grid_points, skip_first=0)
    T, wT = ds["T"], ds["wT"]
    Nz, Nt = size(T)
    Nt = min(Nt, 500+skip_first)

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    inputs = cat([coarse_grain(T[:, n], grid_points) for n in 1+skip_first:Nt]..., dims=2)
    outputs = cat([coarse_grain(wT[:, n], grid_points) for n in 1+skip_first:Nt]..., dims=2)

    μ_T, σ_T = mean(inputs), std(inputs)
    μ_wT, σ_wT = mean(outputs), std(outputs)

    standardize_T(x) = (x - μ_T) / σ_T
    standardize⁻¹_T(y) = σ_T * y + μ_T
    standardize_wT(x) = (x - μ_wT) / σ_wT
    standardize⁻¹_wT(y) = σ_wT * y + μ_wT

    inputs = standardize_T.(inputs)
    outputs = standardize_wT.(outputs)

    training_data = [(inputs[:, n], outputs[:, n]) for n in 1:Nt-skip_first] |> shuffle

    standardization = (
        T = (μ=μ_T, σ=σ_T, standardize=standardize_T, standardize⁻¹=standardize⁻¹_T),
        wT = (μ=μ_wT, σ=σ_wT, standardize=standardize_wT, standardize⁻¹=standardize⁻¹_wT)
    )

    return training_data, standardization
end

function free_convection_time_step_training_data(ds, standardization; grid_points, future_time_steps=1)
    T = ds["T"]
    Nz, Nt = size(T)
    S_T = standardization.T.standardize

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    if future_time_steps == 1
        training_data =
            [(coarse_grain(T[:, n],   grid_points) .|> S_T,
              coarse_grain(T[:, n+1], grid_points) .|> S_T)
             for n in 1:Nt-1]
    else
        training_data =
            [(coarse_grain(T[:, n], grid_points) .|> S_T,
              cat((coarse_grain(T[:, k], grid_points) for k in n+1:n+future_time_steps)..., dims=2) .|> S_T)
             for n in 1:Nt-future_time_steps]
    end

    return training_data
end

function train_on_heat_flux!(NN, training_data, optimizer; epochs=1)
    loss(T, wT) = Flux.mse(NN(T), wT)

    function cb()
        Σloss = [loss(T, wT) for (T, wT) in training_data] |> sum
        @info @sprintf("Training on heat flux... Σloss = %e", Σloss)
        return loss
    end

    NN_params = Flux.params(NN)
    for e in 1:epochs
        @info "Training on heat flux with $(typeof(optimizer))(η=$(optimizer.eta)) [epoch $e/$epochs]..."
        Flux.train!(loss, NN_params, training_data, optimizer, cb=Flux.throttle(cb, 2))
    end

    return nothing
end

function animate_learned_heat_flux(ds, NN, standardization; grid_points, frameskip=1, fps=15)
    T, wT, z = ds["T"], ds["wT"], ds["zC"]
    Nz, Nt = size(T)
    z_coarse = coarse_grain(z, grid_points)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting learned heat flux [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(wT[:, n], z, linewidth=2, xlim=(-1e-5, 3e-5), ylim=(-100, 0),
             label="Oceananigans wT", xlabel="Heat flux", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        S_T, S⁻¹_wT = standardization.T.standardize, standardization.wT.standardize⁻¹

        if NN isa Flux.Chain
            wT_NN = coarse_grain(T[:, n], grid_points) .|> S_T |> NN .|> S⁻¹_wT
            plot!(wT_NN, z_coarse, linewidth=2, label="Neural network")
        elseif NN isa DiffEqFlux.NeuralODE
            wT_NN = coarse_grain(T[:, n], grid_points) .|> S_T
            wT_NN = npde.model(wT_NN, npde.p) .|> S⁻¹_wT
            plot!(wT_NN, z_coarse, linewidth=2, label="Neural network")
        end
    end

    filename = "free_convection_learned_heat_flux.mp4"
    @info "Saving $filename"
    mp4(anim, filename, fps=fps)

    return nothing
end

function construct_neural_pde(NN, ds, standardization; grid_points, Δt, time_steps)
    Nz = grid_points
    zC = coarse_grain(ds["zC"], Nz)
    ΔzC = diff(zC)[1]

    # Computes the derivative from cell center to cell (f)aces
    Dzᶠ = 1/ΔzC * Tridiagonal(-ones(Nz-1), ones(Nz), zeros(Nz-1))

    # Set up neural network for PDE
    # ∂T/dt = - ∂z(wT) + ...
    σ_wT = standardization.wT.σ
    NN_∂T∂t = FastChain(NN.layers..., (wT,_) -> -Dzᶠ * wT)

    # Set up neural differential equation
    tspan = (0.0, time_steps * Δt)
    tsteps = range(tspan[1], tspan[2], length = time_steps+1)
    return NeuralODE(NN_∂T∂t, tspan, ROCK4(), reltol=1e-3, saveat=tsteps)
end

function train_free_convection_neural_pde!(npde, training_data, optimizers; epochs=1)
    time_steps = length(npde.kwargs[:saveat])
    sol_correct = cat([training_data[n][1] for n in 1:time_steps]..., dims=2)

    T₀ = training_data[1][1]
    loss(θ) = Flux.mse(Array(npde(T₀, θ)), sol_correct)

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
    z_coarse = coarse_grain(z, grid_points)

    S_T, S⁻¹_T = standardization.T.standardize, standardization.T.standardize⁻¹

    T₀_NN = coarse_grain(T[:, 1], grid_points) .|> S_T
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
##### Script starts here
#####

ds = NCDataset("free_convection_horizontal_averages.nc")

# Should not have saved constant units as strings...
nc_constant(ds, attr) = parse(Float64, ds.attrib[attr] |> split |> first)

const Q  = nc_constant(ds, "Heat flux")
const ρ₀ = nc_constant(ds, "Reference density")
const cₚ = nc_constant(ds, "Specific_heat_capacity")

Nz = 16  # Number of grid points for the neural PDE.

skip_first = 5
future_time_steps = 1

# animate_variable(ds, "T", grid_points=16, xlabel="Temperature T (°C)", xlim=(19, 20), frameskip=5)
# animate_variable(ds, "wT", grid_points=16, xlabel="Heat flux", xlim=(-1e-5, 3e-5), frameskip=5)

training_data_heat_flux, standardization =
    free_convection_heat_flux_training_data(ds, grid_points=Nz, skip_first=skip_first)

if training_data_heat_flux isa DataLoader
    @info "Heat flux training data contains $(training_data_heat_flux.nobs) pairs (batchsize=$(training_data_heat_flux.batchsize))."
else
    @info "Heat flux training data contains $(length(training_data_heat_flux)) pairs."
end

NN_heat_flux_filename = "NN_heat_flux.bson"
if isfile(NN_heat_flux_filename)
    @info "Loading $NN_heat_flux_filename..."
    BSON.@load NN_heat_flux_filename NN
else
    top_flux = standardization.wT.standardize(Q / (ρ₀ * cₚ))
    bot_flux = standardization.wT.standardize(0.0)
    NN = free_convection_neural_pde_architecture(Nz, top_flux=top_flux, bottom_flux=bot_flux)

    for opt in [ADAM(1e-2), Descent(1e-2), Descent(1e-3)]
        train_on_heat_flux!(NN, training_data_heat_flux, opt, epochs=10)
    end

    @info "Saving $NN_heat_flux_filename..."
    BSON.@save NN_heat_flux_filename NN

    animate_learned_heat_flux(ds, NN, standardization, grid_points=Nz, frameskip=5, fps=15)
end

training_data_time_step =
    free_convection_time_step_training_data(ds, standardization, grid_points=Nz, future_time_steps=future_time_steps)

if training_data_time_step isa DataLoader
    @info "Time step training data contains $(training_data_heat_flux.nobs) pairs (batchsize=$(training_data_heat_flux.batchsize))."
else
    @info "Time step training data contains $(length(training_data_heat_flux)) pairs."
end

FastLayer(layer) = layer

function FastLayer(layer::Dense)
    N_out, N_in = size(layer.W)
    return FastDense(N_in, N_out, layer.σ, initW=(_,_)->layer.W, initb=_->layer.b)
end

FastChain(NN::Chain) = FastChain([FastLayer(layer) for layer in NN]...)

NN_fast = FastChain(NN)
npde = construct_neural_pde(NN_fast, ds, standardization, grid_points=Nz, Δt=1.0, time_steps=50)

for (Nt, epochs) in zip((50, 100, 200, 325, 500, 750), (5, 4, 3, 3, 3, 3))
    global npde
    new_npde = construct_neural_pde(NN_fast, ds, standardization, grid_points=Nz, Δt=1.0, time_steps=Nt)
    new_npde.p .= npde.p; npde = new_npde; # Keep using the same weights/parameters!
    train_free_convection_neural_pde!(npde, training_data_time_step, [ADAM(1e-3)], epochs=epochs)
end

npde_filename = "free_convection_neural_pde_parameters.bson"
@info "Saving $npde_filename..."
npde_params = npde.p
BSON.@save npde_filename npde_params

animate_learned_free_convection(ds, npde, standardization, grid_points=Nz, skip_first=skip_first)
