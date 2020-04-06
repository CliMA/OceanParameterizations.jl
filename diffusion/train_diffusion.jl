using LinearAlgebra
using DiffEqFlux
using Flux
using Optim
using ClimateSurrogates

include("diffusion_surrogate.jl")

const N = 16  # Number of grid points
const L = 2   # Domain size -L/2 <= x <= L/2
const κ = 1   # Diffusivity
const T = 0.1 # Time span 0 <= T <= 0.1
const Nt = 32 # Number of time snapshots to save

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

#####
##### Train and test a neural differential equation
#####

# optimizers = [Descent(1e-5), LBFGS()]
# diffusion_npde = train_diffusion_neural_pde(training_data, dudt_NN, optimizers)
# test_diffusion_neural_pde(diffusion_npde, solutions)

#####
##### Train and test a Gaussian process
#####

# gp = train_diffusion_gp(training_data)
# test_diffusion_gp(gp, solutions)
