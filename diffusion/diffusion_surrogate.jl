using DifferentialEquations
using DiffEqFlux
using Flux

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
function solve_diffusion_equations(; u₀, N, L, κ, T, Nt)
    Δx = L / N
    x = range(-L/2, L/2, length=N)

    tspan = (0.0, T)
    Δt = (tspan[2] - tspan[1]) / Nt
    t = range(tspan[1], tspan[2], length=Nt)
    
    params = (N=N, Δx=Δx, κ=κ)
    prob = ODEProblem(diffusion!, u₀, tspan, params)
    sol = solve(prob, Tsit5(), saveat=t);
end
