using LinearAlgebra
using DiffEqFlux
using Flux

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
dTdt_NN = Chain(Dense(cr+2,  2cr, tanh),
                Dense(2cr,  cr+2),
                NN -> Dzᶜ * NN + heat_flux)

NN_params = Flux.params(dTdt_NN)

error("Stopping")

#####
##### Pre-train the neural network on (T, wT) data pairs
#####

pre_loss_function(∂zTₙ, wTₙ) = sum(abs2, dTdt_NN(∂zTₙ) .- wTₙ)

popt = ADAM(0.01)

function precb()
    loss = sum(abs2, [pre_loss_function(pre_training_data[i]...) for i in 1:N-1])
    println("loss = $loss")
end

pre_train_epochs = 5
for _ in 1:pre_train_epochs
    Flux.train!(pre_loss_function, NN_params, pre_training_data, popt, cb = Flux.throttle(precb, 5))
end

#####
##### Define loss function
#####

tspan = (0.0, 600.0)  # 10 minutes
neural_pde_prediction(T₀) = neural_ode(dTdt_NN, T₀, tspan, Tsit5(), reltol=1e-4, save_start=false, saveat=tspan[2])

loss_function(Tₙ, Tₙ₊₁) = sum(abs2, Tₙ₊₁ .- neural_pde_prediction(Tₙ))

#####
##### Choose optimization algorithm
#####

opt = ADAM(0.1)

#####
##### Callback function to observe training.
#####

function cb()
    train_loss = sum([loss_function(Tₙ[:, i], Tₙ₊₁[:, i]) for i in 1:N])

    nn_pred = neural_ode(dTdt_NN, Tₙ[:, 1], (t[1], t[N]), Tsit5(), saveat=t[1:N], reltol=1e-4) |> Flux.data
    test_loss = sum(abs2, T_cr[:, 1:N] .- nn_pred)

    println("train_loss = $train_loss, test_loss = $test_loss")
    return train_loss
end

cb()

#####
##### Train!
#####

epochs = 10
best_loss = Inf
last_improvement = 0

for epoch_idx in 1:epochs
    global best_loss, last_improvement

    @info "Epoch $epoch_idx"
    Flux.train!(loss_function, NN_params, training_data, opt, cb=cb) # cb=Flux.throttle(cb, 10))

    loss = cb()

    if loss <= best_loss
        @info("Record low loss! Saving neural network out to dTdt_NN.bson")
        BSON.@save "dTdt_NN.bson" dTdt_NN
        best_loss = loss
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 2 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 2 && opt.eta > 1e-6
        opt.eta /= 2.0
        @warn("Haven't improved in a while, dropping learning rate to $(opt.eta)")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end
end
