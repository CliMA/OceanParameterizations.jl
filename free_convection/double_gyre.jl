# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using LinearAlgebra
using Statistics
using Printf

using Oceananigans
using Oceananigans.Units

using Dates: now, Second, format
using Oceananigans.BuoyancyModels: g_Earth, ∂z_b
using Oceananigans.Diagnostics: accurate_cell_advection_timescale
using Oceananigans.Simulations: get_Δt

## Progress

mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9
    progress = sim.model.clock.time / sim.stop_time
    ETA = (1 - progress) / progress * sim.run_time
    ETA_datetime = now() + Second(round(Int, ETA))

    @info @sprintf("[%06.2f%%] Time: %s, iteration: %d, max(|u⃗|): (%.2e, %.2e) m/s, T: (min=%.2f, mean=%.2f, max=%.2f), CFL: %.2e",
                   100 * progress,
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   minimum(sim.model.tracers.T),
                   mean(sim.model.tracers.T),
                   maximum(sim.model.tracers.T),
                   sim.parameters.cfl(sim.model))

    @info @sprintf("           ETA: %s (%s), Δ(wall time): %s / iteration",
                   format(ETA_datetime, "yyyy-mm-dd HH:MM:SS"),
                   prettytime(ETA),
                   prettytime(wall_time / sim.iteration_interval))

    p.interval_start_time = time_ns()

    return nothing
end

## Convective adjustment

function convective_adjustment!(model, Δt, K)
    grid = model.grid
    buoyancy = model.buoyancy
    tracers = model.tracers
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δz = model.grid.Δz
    T = model.tracers.T

    κ = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        κ[i, j, k] = ∂z_b(i, j, k, grid, buoyancy, tracers) < 0 ? K : 0
    end

    T_interior = interior(T)
    Tⁿ⁺¹ = zeros(Nx, Ny, Nz)

    for j in 1:Ny, i in 1:Nx
        ld = [-Δt/Δz^2 * κ[i, j, k]   for k in 2:Nz]
        ud = [-Δt/Δz^2 * κ[i, j, k+1] for k in 1:Nz-1]

        d = zeros(Nz)
        for k in 1:Nz-1
            d[k] = 1 + Δt/Δz^2 * (κ[i, j, k] + κ[i, j, k+1])
        end
        d[Nz] = 1 + Δt/Δz^2 * κ[i, j, Nz]

        𝓛 = Tridiagonal(ld, d, ud)

        Tⁿ⁺¹[i, j, :] .= 𝓛 \ T_interior[i, j, :]
    end

    set!(model, T=Tⁿ⁺¹)

    return nothing
end


@info "Grid setup..."

grid = RegularLatitudeLongitudeGrid(size=(60, 60, 32), latitude=(15, 75), longitude=(0, 60), z=(-2kilometers, 0))


## Boundary conditions

@info "Boundary conditions setup..."

@inline wind_stress(λ, φ, t, p) = - p.τ * cos(2π * (φ - p.φ₀) / p.Lφ)
@inline u_bottom_stress(λ, φ, t, u, p) = - p.μ * p.H * u
@inline v_bottom_stress(λ, φ, t, v, p) = - p.μ * p.H * v

wind_stress_params = (τ=1e-4, Lφ=grid.Ly, φ₀=15)
wind_stress_bc = FluxBoundaryCondition(wind_stress, parameters=wind_stress_params)

bottom_stress_params = (μ=1/30day, H=grid.Lz)
u_bottom_stress_bc = FluxBoundaryCondition(u_bottom_stress, field_dependencies=:u, parameters=bottom_stress_params)
v_bottom_stress_bc = FluxBoundaryCondition(v_bottom_stress, field_dependencies=:v, parameters=bottom_stress_params)

no_slip = ValueBoundaryCondition(0)

u_bcs = UVelocityBoundaryConditions(grid,
       top = wind_stress_bc,
    bottom = u_bottom_stress_bc,
    #  north = no_slip,
    #  south = no_slip
)

v_bcs = VVelocityBoundaryConditions(grid,
    #   east = no_slip,
    #   west = no_slip,
    bottom = v_bottom_stress_bc
)

@inline T_reference(φ, p) = p.T_min + p.ΔT / p.Lφ * (φ - p.φ₀)
@inline temperature_flux(λ, φ, t, T, p) = @inbounds - p.μ * (T - T_reference(φ, p))

T_min, T_max = 0, 30
temperature_flux_params = (T_min=T_min, T_max=T_max, T_mid=(T_min+T_max)/2, ΔT=T_max-T_min, μ=1/day, Lφ=grid.Ly, φ₀=15)
temperature_flux_bc = FluxBoundaryCondition(temperature_flux, field_dependencies=:T, parameters=temperature_flux_params)

T_bcs = TracerBoundaryConditions(grid,
    bottom = ValueBoundaryCondition(T_min),
    top = temperature_flux_bc
)


## Model setup

@info "Model setup..."

model = HydrostaticFreeSurfaceModel(
           architecture = CPU(),
                   grid = grid,
     momentum_advection = VectorInvariant(),
         # free_surface = ExplicitFreeSurface(gravitational_acceleration=0.01g_Earth),
           free_surface = ImplicitFreeSurface(gravitational_acceleration=g_Earth, tolerance=1e-6),
               coriolis = HydrostaticSphericalCoriolis(scheme=VectorInvariantEnstrophyConserving()),
                closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=5000, νz=1e-2, κh=1000, κz=1e-2),
    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs)
)

## Initial condition

@info "Setting initial conditions..."

# a stable density gradient with random noise superposed.
T₀(λ, φ, z) = temperature_flux_params.T_min + temperature_flux_params.ΔT/2 * (1 + z / grid.Lz)
set!(model, T=T₀)

## Simulation setup

@info "Setting up simulation..."

Δt = model.free_surface isa ImplicitFreeSurface ? 1hours : 10minutes

g = model.free_surface.gravitational_acceleration
H = grid.Lz
gravity_wave_speed = √(g * H)
min_spacing = Oceananigans.Operators.Δxᶠᶠᵃ(1, grid.Ny, 1, grid)
wave_propagation_time_scale = min_spacing / gravity_wave_speed
gravity_wave_cfl = Δt / wave_propagation_time_scale
@info @sprintf("Gravity wave CFL = %.4f", gravity_wave_cfl)

cfl = CFL(Δt, accurate_cell_advection_timescale)

simulation = Simulation(model,
                    Δt = Δt,
             stop_time = 90days,
    iteration_interval = 1,
              progress = Progress(time_ns()),
            parameters = (; cfl)
)

## Set up output writers

@info "Setting up output writers..."

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
            schedule = TimeInterval(1day),
              prefix = "double_gyre",
        field_slicer = FieldSlicer(with_halos=true),
               force = true)

## Running the simulation

@info "Running simulation..."
run!(simulation)
