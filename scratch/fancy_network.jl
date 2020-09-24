using Statistics
using Flux
using DifferentialEquations
using DiffEqSensitivity
using DiffEqFlux
using DiffEqFlux: basic_tgrad

N = 4
n_data = 5

# input ∈ ℝ⁴ + 2 parameters 
# output ∈ ℝ⁵

inputs = [(x=rand(N), a=i, b=-i) for i in 1:N] 
outputs = [cat(i, rand(N-1), -i, dims=1) for i in 1:N]
data = [(i, o) for (i, o) in zip(inputs, outputs)]

# Flux.jl

NN = Chain(Dense(N, 2N), Dense(2N, N-1))

function NN_fancy(x, a, b)
    ϕ = NN(x)
    return cat(a, ϕ, b, dims=1)
end

loss(input, y) = Flux.mse(NN_fancy(input.x, input.a, input.b), y)

function cb()
    μ_loss = mean(loss(input, y) for (input, y) in data)
    @info "μ_loss = $μ_loss"
end

Flux.train!(loss, Flux.params(NN), data, ADAM(), cb=cb)

# DiffEqFlux.jl

p, reconstruct = Flux.destructure(NN)

function NN_pde(NN, x, a, b)
    x′ = 2x
    ϕ = NN(x′)
    ϕ′ = ϕ / 7
    y = cat(a, ϕ′, b, dims=1)
    return diff(y)
end

function loss(θ)
    NN = reconstruct(θ)
    μ_loss = mean(Flux.mse(NN_pde(NN, input.x, input.a, input.b), y[1:end-1]) for (input, y) in data)
    return μ_loss
end

function cb(θ, μ_loss)
    @info "μ_loss = $μ_loss"
    return false
end

res = DiffEqFlux.sciml_train(loss, p, ADAM(), cb=cb, maxiters=10, save_best=true)
p = res.minimizer
NN = reconstruct(p)

# NeuralODE

function RHS(x, p)
    a, b = p[end-1], p[end]
    x′ = 2x
    ϕ = NN(x′)
    ϕ′ = ϕ / 7
    y = cat(a, ϕ′, b, dims=1)
    return diff(y)
end

npde = NeuralODE(FastChain(RHS), (0.0, 1.0), ROCK4(), p=p, reltol=1e-3, saveat=0:0.1:1)

function (npde::typeof(npde))(u₀, p)
    dudt_(u, p, t) = npde.re(p[1:end-2])(u, p)
    ff = ODEFunction{false}(dudt_, tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, u₀, getfield(npde, :tspan), p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob, npde.args...; sense=sense, npde.kwargs...)
end

pde_data = randn(4, 11)
u₀ = pde_data[:, 1]

params = cat(p, 1, -2, dims=1)

loss(θ) = Flux.mse(Array(npde(u₀, θ)), pde_data)

function cb(θ, μ_loss)
    @info "a=$(θ[end-1]), b=$(θ[end])"
    @info "μ_loss = $μ_loss"
    return false
end
    
res = DiffEqFlux.sciml_train(loss, params, ADAM(), cb=cb, maxiters=10, save_best=true)
