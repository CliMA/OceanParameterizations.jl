using Printf
using LinearAlgebra

using Plots
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.AbstractOperations
using Oceananigans.Utils

ENV["GKSwstype"] = "100"

#####
##### Convective adjustment
#####

function convective_adjustment!(model, Δt, K)
    Nz, Δz = model.grid.Nz, model.grid.Δz
    T = model.tracers.T
    
    ∂T∂z = ComputedField(@at (Cell, Cell, Cell) ∂z(T))
    compute!(∂T∂z)

    κ = zeros(Nz)
    for i in 1:Nz
        κ[i] = ∂T∂z[1, 1, i] < 0 ? K : 0
    end

    ld = [-Δt/Δz^2 * κ[i]   for i in 2:Nz]
    ud = [-Δt/Δz^2 * κ[i+1] for i in 1:Nz-1]

    d = zeros(Nz)
    for i in 1:Nz-1
        d[i] = 1 + Δt/Δz^2 * (κ[i] + κ[i+1])
    end
    d[Nz] = 1 + Δt/Δz^2 * κ[Nz]

    𝓛 = Tridiagonal(ld, d, ud)
    
    T′ = 𝓛 \ interior(T)[:]
    set!(model, T=reshape(T′, (1, 1, Nz)))

    return nothing
end

#####
##### Run free convection
#####

ρ₀ = 1027
cₚ = 4000
Q  = 100
∂T₀∂z = 0.01
K = 1000

grid = RegularCartesianGrid(size=(1, 1, 32), extent=(1, 1, 100))

T_bc_top = BoundaryCondition(Flux, Q / (ρ₀ * cₚ))
T_bc_bot = BoundaryCondition(Gradient, ∂T₀∂z)
T_bcs = TracerBoundaryConditions(grid, top=T_bc_top)

model = IncompressibleModel(grid=grid, boundary_conditions=(T=T_bcs,))

T₀(x, y, z) = 20 + 0.01z
set!(model, T=T₀)

Δt = 10minute
stop_time = 1day
Nt = Int(stop_time / Δt)

anim = @animate for n in 1:Nt
    time_step!(model, Δt)
    convective_adjustment!(model, Δt, K)

    time_str = @sprintf("%.2f days", model.clock.time / day)
    @info "t = $time_str"

    T = interior(model.tracers.T)[:]
    z = znodes(Cell, grid)
    plot(T, z, linewidth=2, xlim=(19, 20), ylim=(-100, 0),
         xlabel="Temperature (°C)", ylabel="Depth z (meters)",
         title="Free convection: $time_str", legend=:bottomright, show=false)
end

mp4(anim, "oceananigans_free_convection.mp4", fps=30)
