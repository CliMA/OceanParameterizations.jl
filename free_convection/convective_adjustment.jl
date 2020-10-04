using Printf
using LinearAlgebra

using DifferentialEquations
using Plots

ENV["GKSwstype"] = "100"

""" Returns a discrete 1D derivative operator for cell center to cell (f)aces. """
function Dᶠ(N, Δ)
    D = zeros(N, N+1)
    for k in 1:N
        D[k, k]   = -1.0
        D[k, k+1] =  1.0
    end
    D = 1/Δ * D
    return D
end

""" Returns a discrete 1D derivative operator for cell faces to cell (c)enters. """
function Dᶜ(N, Δ)
    D = zeros(N+1, N)
    for k in 2:N
        D[k, k-1] = -1.0
        D[k, k]   =  1.0
    end
    D = 1/Δ * D
    return D
end

function ℑᶠ(N, Δ)
    ℑ = zeros(N, N+1)
    for k in 1:N
        ℑ[k, k]   = 0.5
        ℑ[k, k+1] = 0.5
    end
    return ℑ
end

function ℑᶜ(N, Δ)
    ℑ = zeros(N+1, N)
    for k in 2:N
        ℑ[k, k-1] = 0.5
        ℑ[k, k]   = 0.5
    end
    return ℑ
end

function convective_adjustment(T, p, t)
    N, κ, Qₛ = p.N, p.κ, p.Qₛ
    Dzᶠ, Dzᶜ = p.Dzᶠ, p.Dzᶜ

    ∂T∂z = Dzᶜ * T

    wT = @. min(0, κ * ∂T∂z)
    wT[N+1] += Qₛ

    return Dzᶠ * wT
end

function surface_flux(T, p, t)
    N, Qₛ, Dzᶠ = p.N, p.Qₛ, p.Dzᶠ

    wT = zeros(N+1)
    wT[N+1] += Qₛ

    return Dzᶠ * wT
end

N = 32
L = 100
Δ = L/N
z = range(Δ/2, L-Δ/2, length=N)

Dzᶠ = Dᶠ(N, Δ)
Dzᶜ = Dᶜ(N, Δ)
ℑzᶠ = ℑᶠ(N, Δ)
ℑzᶜ = ℑᶜ(N, Δ)

ρ₀ = 1027
cₚ = 4000
Q  = -100 / (ρ₀ * cₚ)

T₀ = [19 + zₖ/L for zₖ in z]

day = 86400
tspan = (0.0, 8day)
tsave = range(tspan...; length=101)

params = (N=N, κ=1000, Qₛ=Q, Dzᶠ=Dzᶠ, Dzᶜ=Dᶜ(N, Δ))

# prob = ODEProblem(convective_adjustment, T₀, tspan, params)
# @time sol = solve(prob, Rodas5(), reltol=1e-5, saveat=tsave, progress=true)
#
# anim = @animate for n in 1:length(sol)
#     @info "Frame $n/$(length(sol))"
#     time_str = @sprintf("%.2f days", sol.t[n] / day)
#
#     plot(sol[n], z, linewidth=2, xlim=(19, 20), ylim=(0, 100),
#          xlabel="Temperature (°C)", ylabel="Depth z (meters)",
#          title="Convective adjustment: $time_str", legend=:bottomright, show=false)
# end
#
# mp4(anim, "convective_adjustment.mp4", fps=15)

function convective_adjustment!(integrator)
    T = integrator.u
    ∂T∂z = Dzᶜ * T
    ℑ∂T∂z = ℑzᶠ * ∂T∂z

    κ = zeros(N)
    for j in 1:N
        κ[j] = ℑ∂T∂z[j] < 0 ? K : 0
    end

    ld = [-Δt/Δz^2 * κ[j]   for j in 2:N]
    ud = [-Δt/Δz^2 * κ[j+1] for j in 1:N-1]

    d = zeros(N)
    for j in 1:N-1
        d[j] = 1 + Δt/Δz^2 * (κ[j] + κ[j+1])
    end
    d[N] = 1 + Δt/Δz^2 * κ[N]

    𝓛 = Tridiagonal(ld, d, ud)
    T .= 𝓛 \ T

    return nothing
end

Δz = Δ
Δt = 3600.0
K  = 1000
time_steps = 100

prob = ODEProblem(surface_flux, T₀, tspan, params)
integrator = init(prob, Tsit5(), adaptive=false, dt=Δt)

anim = @animate for n in 1:time_steps
    step!(integrator)
    convective_adjustment!(integrator)

    @info "frame $n/$time_steps"
    time_str = @sprintf("%.2f days", integrator.t / day)
    plot(integrator.sol[n], z, linewidth=2, xlim=(19, 20), ylim=(0, 100),
         xlabel="Temperature (°C)", ylabel="Depth z (meters)",
         title="Convective adjustment: $time_str", legend=:bottomright, show=false)
end
 
mp4(anim, "convective_adjustment.mp4", fps=30)
