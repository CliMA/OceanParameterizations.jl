using DiffEqFlux
using Flux
using Optim
using ClimateSurrogates

include("diffusion_surrogate.jl")

#####
##### Training and testing initial condition functions
#####

u₀_Gaussian(x) = exp(-50x^2)
u₀_quadratic(x) = 1 - x^2
u₀_sin(x) = 1 + sin(2π * x)
u₀_cos(x) = 1 - cos(2π * x)
u₀_shifted_sin(x) = 1 + sin(π * x + π/3)
u₀_shifted_cos(x) = 1 - cos(π * x - π/6)
u₀_zero(x) = 0.0
u₀_one(x) = 1.0

function_name(::typeof(u₀_Gaussian)) = "Gaussian"
function_name(::typeof(u₀_quadratic)) = "quadratic"
function_name(::typeof(u₀_sin))  = "sin"
function_name(::typeof(u₀_cos))  = "cos"
function_name(::typeof(u₀_shifted_sin))  = "shifted_sin"
function_name(::typeof(u₀_shifted_cos))  = "shifted_cos"
function_name(::typeof(u₀_zero)) = "zero"
function_name(::typeof(u₀_one)) = "one"

training_functions = (u₀_Gaussian, u₀_cos, u₀_shifted_sin, u₀_zero)
testing_functions = (u₀_quadratic, u₀_shifted_cos, u₀_sin, u₀_one)

#####
##### Generate truth solutions for training and testing
#####

const N = 16
const Nt = 32

solutions = Dict()
training_data = []

for u₀ in (training_functions..., testing_functions...)
    sol, x, Δt = solve_diffusion_equation(u₀=u₀, N=N, L=2, κ=1, T=0.1, Nt=Nt)
    solutions[function_name(u₀)] = sol

    u₀ in training_functions && append!(training_data, generate_training_data(sol))

    fname = "diffusing_$(function_name(u₀)).mp4"
    animate_solution(x, sol, filename=fname)
end

#####
##### Set up neural differential equation
#####

dudt_NN = FastChain(FastDense(N, N))
#  dudt_NN = FastChain(FastDense(N, 100, tanh),
#                      FastDense(100, N))

Δt = 0.1 / Nt
tspan_npde = (0.0, Δt)
diffusion_npde = NeuralODE(dudt_NN, tspan_npde, Tsit5(), reltol=1e-4, saveat=[Δt])

loss_function(θ, uₙ, uₙ₊₁) = Flux.mse(uₙ₊₁, diffusion_npde(uₙ, θ)) ./ (sum(abs2, uₙ₊₁) + 1e-6)
training_loss(θ, data) = sum([loss_function(θ, data[i]...) for i in 1:length(data)])

function cb(θ, args...)
    println("train_loss = $(training_loss(θ, training_data))")
    return false
end

#####
##### Train!
#####

for opt in [ADAM(1e-3), ADAM(1e-4), LBFGS()]
    @info "Training with optimizer: $(typeof(opt))..."
    if opt isa ADAM
        epochs = 10
        for e in 1:epochs
            @info "Training with optimizer: $(typeof(opt)) epoch $e..."
            res = DiffEqFlux.sciml_train(loss_function, diffusion_npde.p, opt, training_data, cb=cb)
            diffusion_npde.p .= res.minimizer
        end
    else
        full_loss(θ) = training_loss(θ, training_data)
        res = DiffEqFlux.sciml_train(full_loss, diffusion_npde.p, opt, cb=cb, maxiters=1000)
        display(res)
        diffusion_npde.p .= res.minimizer
    end
end

#####
##### Test!
#####

x = range(-1, 1, length=N)
for (name, sol) in solutions
    u_NN = animate_neural_de_test(sol, diffusion_npde, x, filename="test_$name.mp4")
    plot_conservation(u_NN, sol.t, filename="conservation_$name.mp4")
end

