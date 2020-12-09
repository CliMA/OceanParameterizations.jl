using LinearAlgebra
using DiffEqFlux
using Flux
using Optim
using OceanParameterizations

include("diffusion.jl")
include("diffusion_npde.jl")
include("diffusion_gp.jl")
include("diffusion_test.jl")

N = 16  # Number of grid points
L = 2   # Domain size -L/2 <= x <= L/2
κ = 1.5 # Diffusivity
T = 0.1 # Time span 0 <= T <= 0.1
Nt = 32 # Number of time snapshots to save
Δt = T/Nt

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

solutions, training_solutions, testing_solutions, training_data, testing_data =
    generate_solutions(training_functions, testing_functions; N=N, L=L, κ=κ, T=T, Nt=Nt, animate=false)

x_train = [data[1] for data in training_data]
y_train = [data[2] for data in training_data]

#####
##### Train and test a neural differential equation
#####

dudt_NN = generate_neural_pde_architecture(N, κ, type=:conservative_feed_forward)
optimizers = [Descent(1e-5)]
diffusion_npde = train_diffusion_neural_pde(training_data, dudt_NN, optimizers, Δt)
test_diffusion_neural_pde(diffusion_npde, solutions)

#####
##### Train and test a Gaussian process
#####

# gp = train_diffusion_gp(training_data)
# test_diffusion_gp(gp, solutions)
