using DiffEqFlux
using Flux
using Optim

include("diffusion_surrogate.jl")

u₀_Gaussian(x) = exp(-50x^2)

const N = 16
const Nt = 32

sol, x, Δt = solve_diffusion_equation(u₀=u₀_Gaussian, N=N, L=2, κ=1, T=0.1, Nt=Nt)
training_data = generate_training_data(sol)

animate_solution(x, sol, filename="diffusing_gaussian.mp4")

dudt_NN = FastChain(FastDense(N, 100, tanh),
                    FastDense(100, N))

tspan_npde = (0.0, Δt)
diffusion_npde = NeuralODE(dudt_NN, tspan_npde, Tsit5(), reltol=1e-4, saveat=[Δt])

loss_function(θ, uₙ, uₙ₊₁) = Flux.mse(uₙ₊₁, diffusion_npde(uₙ, θ))

opt = ADAM(1e-2)

function cb(θ, args...)
    train_loss = sum([loss_function(θ, training_data[i]...) for i in 1:Nt-1])
    println("train_loss = $train_loss")
    return false
end

cb(diffusion_npde.p)

DiffEqFlux.sciml_train(loss_function, diffusion_npde.p, LBFGS(), training_data, cb=cb, maxiters=10)
DiffEqFlux.sciml_train(loss_function, diffusion_npde.p, ADAM(1e-2), training_data, cb=cb, maxiters=10)

