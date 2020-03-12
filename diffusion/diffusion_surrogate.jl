using DifferentialEquations
using DiffEqFlux
using Flux
using Plots

# For quick headless plotting without warnings.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

function diffusion!(∂u∂t, u, p, t)
    N, Δx, κ = p.N, p.Δx, p.κ
    @inbounds begin
        ∂u∂t[1] = κ * (u[N] -2u[1] + u[2]) / Δx
        for i in 2:N-1
            ∂u∂t[i] = κ * (u[i-1] -2u[i] + u[i+1]) / Δx
        end
∂u∂t[N] = κ * (u[N-1] -2u[N] + u[1]) / Δx
    end
    return nothing
end

"""
Solve a 1D diffusion equation with initial condition given by `u₀(x)` on a domain -L/2 <= x <= L/2 with diffusivity `κ` using `N` grid points for time 0 <= t <= `T`. A solution with `Nt` outputs will be returned.
"""
function solve_diffusion_equation(; u₀, N, L, κ, T, Nt)
    Δx = L / N
    x = range(-L/2, L/2, length=N)
    ic = u₀.(x)

    tspan = (0.0, T)
    Δt = (tspan[2] - tspan[1]) / Nt
    t = range(tspan[1], tspan[2], length=Nt)

    params = (N=N, Δx=Δx, κ=κ)
    prob = ODEProblem(diffusion!, ic, tspan, params)
    solution = solve(prob, Tsit5(), saveat=t)

    return solution, x, Δt
end

function generate_training_data(solution)
    N, Nt = size(solution)

    uₙ    = zeros(N, Nt-1)
    uₙ₊₁  = zeros(N, Nt-1)

    for i in 1:Nt-1
           uₙ[:, i] .=  sol.u[i]
         uₙ₊₁[:, i] .=  sol.u[i+1]
    end

    training_data = [(uₙ[:, i], uₙ₊₁[:, i]) for i in 1:Nt-1]

    return training_data
end

function animate_solution(x, sol; filename)
    Nt = length(sol)
    anim = @animate for n=1:Nt
        plot(x, sol.u[n], linewidth=2, ylim=(0, 1), label="", show=false)
    end

    @info "Saving $filename..."
    mp4(anim, filename, fps=15)
end

function test_neural_de(sol, nde, x)
    N, Nt = size(sol)

    u_NN = zeros(N, Nt)
    u_NN[:, 1] .= sol.u[1]

    for n in 2:Nt
        sol_NN = nde(u_NN[:, n-1])
        u_NN[:, n] .= sol_NN.u[1]
    end

    anim = @animate for n=1:Nt
        plot(x, sol.u[n],    linewidth=2, ylim=(0, 1), label="Data", show=false)
        plot!(x, u_NN[:, n], linewidth=2, ylim=(0, 1), label="Neural PDE", show=false)
    end

    mp4(anim, "diffusing_gaussian_test.mp4", fps=15)
end

