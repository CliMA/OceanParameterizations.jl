using DifferentialEquations
using DiffEqFlux
using Flux
using Gen
using Plots
using ClimateSurrogates

# For quick headless plotting without warnings.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

#####
##### Generating solutions and training data
#####

function diffusion!(∂u∂t, u, p, t)
    N, Δx, κ = p.N, p.Δx, p.κ
    @inbounds begin
        ∂u∂t[1] = κ * (u[N] -2u[1] + u[2]) / Δx^2
        for i in 2:N-1
            ∂u∂t[i] = κ * (u[i-1] -2u[i] + u[i+1]) / Δx^2
        end
        ∂u∂t[N] = κ * (u[N-1] -2u[N] + u[1]) / Δx^2
    end
    return nothing
end

"""
    solve_diffusion_equation(; u₀, N, L, κ, T, Nt)

Solve a 1D diffusion equation with initial condition given by `u₀(x)` on a domain -L/2 <= x <= L/2 with diffusivity `κ` using `N` grid points for time 0 <= t <= `T`. A solution with `Nt` outputs will be returned.
"""
function solve_diffusion_equation(; u₀, N, L, κ, T, Nt)
    Δx = L / N
    x = range(-L/2, L/2, length=N)
    ic = u₀.(x)

    tspan = (0.0, T)
    Δt = (tspan[2] - tspan[1]) / Nt
    t = range(tspan[1], tspan[2], length=Nt)

    params = (N=N, Δx=Δx, κ=κ, x=x)
    prob = ODEProblem(diffusion!, ic, tspan, params)
    solution = solve(prob, Tsit5(), saveat=t)

    return solution
end

function generate_training_data(sol)
    N, Nt = size(sol)

    uₙ    = zeros(N, Nt-1)
    uₙ₊₁  = zeros(N, Nt-1)

    for i in 1:Nt-1
           uₙ[:, i] .=  sol.u[i]
         uₙ₊₁[:, i] .=  sol.u[i+1]
    end

    training_data = [(uₙ[:, i], uₙ₊₁[:, i]) for i in 1:Nt-1]

    return training_data
end

function generate_solutions(training_functions, testing_functions; N, L, κ, T, Nt, animate=false)
    # Generate truth solutions for training and testing
    solutions = Dict()
    training_solutions = Dict()
    testing_solutions = Dict()
    training_data = []
    testing_data = []

    for u₀ in (training_functions..., testing_functions...)
        sol = solve_diffusion_equation(u₀=u₀, N=N, L=L, κ=κ, T=T, Nt=Nt)
        solutions[function_name(u₀)] = sol

        if u₀ in training_functions
            training_solutions[function_name(u₀)] = sol
            append!(training_data, generate_training_data(sol))
        elseif u₀ in testing_functions
            testing_solutions[function_name(u₀)] = sol
            append!(testing_data, generate_training_data(sol))
        end

        if animate
            fname = "diffusing_$(function_name(u₀)).mp4"
            animate_solution(sol, filename=fname)
        end
    end

    return solutions, training_solutions, testing_solutions, training_data, testing_data
end

function animate_solution(sol; filename, fps=15)
    Nt = length(sol)
    x = sol.prob.p.x

    anim = @animate for n=1:Nt
        plot(x, sol.u[n], linewidth=2, ylim=(0, 2), label="", show=false)
    end

    mp4(anim, filename, fps=fps)
end
