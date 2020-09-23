using Flux
using DiffEqFlux

N = 4
n_data = 5

# input ∈ ℝ⁴
# output ∈ ℝ⁴

data = [(rand(N), rand(N)) for _ in 1:n_data]

# Flux.jl

NN = Chain(Dense(N, 2N), Dense(2N, N))

loss(x, y) = Flux.mse(NN(x), y)
Flux.train!(loss, Flux.params(NN), data, ADAM())

# DiffEqFlux.jl

p, reconstruct = Flux.destructure(NN)

function loss(θ, curdata...)
    NN = re(θ)
    μ_loss = mean(Flux.mse(NN(x), y) for (x, y) in data)
    return μ_loss
end

function cb(θ, μ_loss)
    @info "μ_loss = $μ_loss"
    return false
end

res = DiffEqFlux.sciml_train(loss, p, ADAM(), cb=cb, maxiters=1000, save_best=true)
p = res.minimizer
NN = reconstruct(p)