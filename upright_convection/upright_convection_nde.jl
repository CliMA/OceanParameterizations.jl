using LinearAlgebra
using DifferentialEquations
using DiffEqFlux
using Flux
using Optim

include("upright_convection.jl")

T, zC, t, Nz, Lz, constants, Q, FT, ∂T∂z, wT =
    load_data("free_convection_profiles.jld2", return_wT=true)

Nt, Nz = size(T)
coarse_resolution = cr = 32
T_cr = zeros(cr+2, Nt)
wT_cr = zeros(cr+2, Nt)

zC_cr = coarse_grain(collect(zC), cr)

for n=1:Nt
    T_cr[2:end-1, n] .= coarse_grain(T[n, :], coarse_resolution)
    wT_cr[2:end-1, n] .= coarse_grain(wT[n, :], coarse_resolution)
end

# Fill halo regions to enforce boundary conditions.
T_cr[1,   :] .= T_cr[2,     :]
T_cr[end, :] .= T_cr[end-1, :]

wT_cr[1,   :] .= wT_cr[2,     :]
wT_cr[end, :] .= wT_cr[end-1, :]

#####
##### Generate differentiation matrices
#####

Δz_cr = Lz / cr  # Coarse resolution Δz

# Dzᶠ computes the derivative from cell center to cell (F)aces
Dzᶠ = 1/Δz_cr * Tridiagonal(-ones(cr+1), ones(cr+2), zeros(cr+1))

# Dzᶜ computes the derivative from cell faces to cell (C)enters
Dzᶜ = 1/Δz_cr * Tridiagonal(zeros(cr+1), -ones(cr+2), ones(cr+1))

# Impose boundary condition that derivative goes to zero at top and bottom.
# Dzᶠ[1, 1] = 0
# Dzᶜ[cr, cr] = 0

#####
##### Create training data
#####

Tₙ    = zeros(cr+2, Nt-1)
Tₙ₊₁  = zeros(cr+2, Nt-1)
wTₙ   = zeros(cr+2, Nt-1)
∂zTₙ  = zeros(cr+2, Nt-1)
∂zwTₙ = zeros(cr+2, Nt-1)

for i in 1:Nt-1
       Tₙ[:, i] .=  T_cr[:,   i]
     Tₙ₊₁[:, i] .=  T_cr[:, i+1]
      wTₙ[:, i] .= wT_cr[:,   i]
     ∂zTₙ[:, i] .= Dzᶠ * T_cr[:, i]
    ∂zwTₙ[:, i] .= Dzᶜ * wT_cr[:, i]
end

N_skip = 0  # Skip first N_skip iterations to avoid learning transients?
N = 3  # Number of training data pairs.

# rinds = randperm(Nt-N_skip)[1:N]

# pre_training_data = [(Tₙ[:, i], ∂zwTₙ[:, i]) for i in 1:N]
pre_training_data = [(∂zTₙ[:, i], -wTₙ[:, i]) for i in 1:N]
training_data = [(Tₙ[:, i], Tₙ₊₁[:, i]) for i in 1:N]

#####
##### Create heat flux vector
#####

heat_flux = zeros(cr+2)
ρ₀ = constants.ρ₀
cₚ = constants.cP

heat_flux[2] = Q / (ρ₀ * cₚ * Δz_cr)

#####
##### Create neural network
#####

# Complete black box right-hand-side.
#  dTdt_NN = Chain(Dense(cr+2,  2cr, tanh),
#                  Dense(2cr,  cr+2))

# Use NN to parameterize a diffusivity or κ profile.
#  dTdt_NN = Chain(T -> Dzᶠ*T,
#                Dense(cr+2,  2cr, tanh),
#                Dense(2cr,  cr+2),
#                NNDzT -> Dzᶜ * NNDzT + heat_flux)

# Use NN to parameterize flux.
NN = dTdt_NN = Chain(Dense(cr+2,  2cr, tanh),
                     Dense(2cr,  cr+2),
                     NN -> Dzᶜ * NN + heat_flux)

NN_params = Flux.params(dTdt_NN)

#####
##### Pre-train the neural network on (T, wT) data pairs
#####

pre_loss_function(∂zTₙ, wTₙ) = sum(abs2, dTdt_NN(∂zTₙ) .- wTₙ)

popt = Flux.ADAM(0.01)

function precb()
    loss = sum(abs2, [pre_loss_function(pre_training_data[i]...) for i in 1:N-1])
    println("loss = $loss")
end

pre_train_epochs = 10
for _ in 1:pre_train_epochs
    Flux.train!(pre_loss_function, NN_params, pre_training_data, popt, cb=Flux.throttle(precb, 1))
end

#####
##### Train!
#####

optimizers = [Descent(1e-2), Descent(1e-3)]

Δt = 600.0  # 10 minutes
tspan_npde = (0.0, Δt)
npde = NeuralODE(NN, tspan_npde, Tsit5(), reltol=1e-4, saveat=[Δt])

loss_function(θ, uₙ, uₙ₊₁) = Flux.mse(uₙ₊₁, npde(uₙ, θ)) ./ (sum(abs2, uₙ₊₁) + 1e-6)
training_loss(θ, data) = sum([loss_function(θ, data[i]...) for i in 1:length(data)])

function cb(θ, args...)
    println("train_loss = $(training_loss(θ, training_data))")
    return false
end

# Train!
for opt in optimizers
    @info "Training with optimizer: $(typeof(opt))..."
    if opt isa Optim.AbstractOptimizer
        full_loss(θ) = training_loss(θ, training_data)
        res = DiffEqFlux.sciml_train(full_loss, npde.p, opt, cb=Flux.throttle(cb, 1), maxiters=100)
        display(res)
        npde.p .= res.minimizer
    else
        epochs = 10
        for e in 1:epochs
            @info "Training with optimizer: $(typeof(opt)) epoch $e..."
            res = DiffEqFlux.sciml_train(loss_function, npde.p, opt, training_data, cb=Flux.throttle(cb, 1))
            npde.p .= res.minimizer
        end
    end
end
