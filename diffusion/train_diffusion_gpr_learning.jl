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

const Δt = T / Nt
const x = range(-L/2, L/2, length=N)

function generate_solutions(training_functions, testing_functions)
    # Generate truth solutions for training and testing
    solutions = Dict()
    training_data = []

    for u₀ in (training_functions..., testing_functions...)
        sol = solve_diffusion_equation(u₀=u₀, N=N, L=L, κ=κ, T=T, Nt=Nt)
        solutions[function_name(u₀)] = sol

        u₀ in training_functions && append!(training_data, generate_training_data(sol))

        fname = "diffusing_$(function_name(u₀)).mp4"
        # animate_solution(x, sol, filename=fname)
    end

    return solutions, training_data
end

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
##### Neural network architecture
#####

solutions, training_data = generate_solutions(training_functions, testing_functions)

# Split data as y = GP(x)
x_train = [data[1] for data in training_data]
y_train = [data[2] for data in training_data]

const γ₁ = 0.0001
const σ₁ = 1.0
K(x, y) = σ₁ * exp(-γ₁ * norm(x - y)^2)
gp = construct_gpr(x_train, y_train, K)

function animate_gp_test(sol, gp, x; filename)
    N, Nt = size(sol)

    u_GP = zeros(N, Nt)
    u_GP[:, 1] .= sol.u[1]

    for n in 2:Nt
        u_GP[:, n] .= prediction([u_GP[:, n-1]], gp)
    end

    anim = @animate for n=1:Nt
        plot(x, sol.u[n],    linewidth=2, ylim=(0, 2), label="Data", show=false)
        plot!(x, u_GP[:, n], linewidth=2, ylim=(0, 2), label="GP", show=false)
    end

    mp4(anim, filename, fps=15)

    return u_GP
end

for (name, sol) in solutions
    u_GP = animate_gp_test(sol, gp, x, filename="GP_test_$name.mp4")
end
