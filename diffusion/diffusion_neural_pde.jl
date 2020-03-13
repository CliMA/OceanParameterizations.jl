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

dudt_NN = FastChain(FastDense(N, N))
#  dudt_NN = FastChain(FastDense(N, 100, tanh),
#                      FastDense(100, N))

tspan_npde = (0.0, Δt)
diffusion_npde = NeuralODE(dudt_NN, tspan_npde, Tsit5(), reltol=1e-4, saveat=[Δt])

loss_function(θ, uₙ, uₙ₊₁) = Flux.mse(uₙ₊₁, diffusion_npde(uₙ, θ)) ./ sum(abs2, uₙ₊₁)
training_loss(θ, data) = sum([loss_function(θ, data[i]...) for i in 1:length(data)])

function cb(θ, args...)
    println("train_loss = $(training_loss(θ, training_data))")
    return false
end

for opt in [ADAM(1e-2), ADAM(1e-3), LBFGS()]
    @info "Training with optimizer: $(typeof(opt))..."
    if opt isa ADAM
        epochs = 10
        for e in 1:epochs
            @info "Training with optimizer: $(typeof(opt)) epoch $e..."
            res = DiffEqFlux.sciml_train(loss_function, diffusion_npde.p, opt, training_data, cb=cb)
            diffusion_npde.p .= res.minimizer
        end
    else
        @info "Training with optimizer: $(typeof(opt))..."
        full_loss(θ) = training_loss(θ, training_data)
        res = DiffEqFlux.sciml_train(full_loss, diffusion_npde.p, opt, cb=cb)
        display(res)
        diffusion_npde.p .= res.minimizer
    end
end

test_neural_de(sol, diffusion_npde, x)
