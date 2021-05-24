# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using LinearAlgebra
using Statistics
using Printf

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.Utils

using Oceananigans.Simulations: get_Δt

## Convective adjustment

function convective_adjustment!(model, Δt, K)
    grid = model.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δz = model.grid.Δz
    T = model.tracers.T

    ∂T∂z = ComputedField(@at (Center, Center, Center) ∂z(T))
    compute!(∂T∂z)

    κ = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        κ[i, j, k] = ∂T∂z[i, j, k] < 0 ? K : 0
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

## Grid setup

@info "Grid setup..."

km = kilometers
topo = (Bounded, Bounded, Bounded)
domain = (x=(-2000km, 2000km), y=(-3000km, 3000km), z=(-2km, 0))
grid = RegularCartesianGrid(topology=topo, size=(96, 96, 32); domain...)

## Boundary conditions

@info "Boundary conditions setup..."

@inline wind_stress(x, y, t, p) = - p.τ * cos(2π * y / p.L)
@inline u_bottom_stress(x, y, t, u, p) = - p.μ * p.H * u
@inline v_bottom_stress(x, y, t, v, p) = - p.μ * p.H * v

wind_stress_params = (τ=1e-4, L=grid.Ly)
wind_stress_bc = FluxBoundaryCondition(wind_stress, parameters=wind_stress_params)

bottom_stress_params = (μ=1/30day, H=grid.Lz)
u_bottom_stress_bc = FluxBoundaryCondition(u_bottom_stress, field_dependencies=:u, parameters=bottom_stress_params)
v_bottom_stress_bc = FluxBoundaryCondition(v_bottom_stress, field_dependencies=:v, parameters=bottom_stress_params)

no_slip = ValueBoundaryCondition(0)

u_bcs = UVelocityBoundaryConditions(grid,
       top = wind_stress_bc,
    bottom = u_bottom_stress_bc,
     north = no_slip,
     south = no_slip
)

v_bcs = VVelocityBoundaryConditions(grid,
      east = no_slip,
      west = no_slip,
    bottom = v_bottom_stress_bc
)

w_bcs = WVelocityBoundaryConditions(grid,
    north = no_slip,
    south = no_slip,
     east = no_slip,
     west = no_slip
)

@inline T_reference(y, p) = p.T_mid + p.ΔT / p.Ly * y
@inline temperature_flux(x, y, t, T, p) = @inbounds - p.μ * (T - T_reference(y, p))

T_min, T_max = 0, 30
temperature_flux_params = (T_min=T_min, T_max=T_max, T_mid=(T_min+T_max)/2, ΔT=T_max-T_min, μ=1/day, Ly=grid.Ly)
temperature_flux_bc = FluxBoundaryCondition(temperature_flux, field_dependencies=:T, parameters=temperature_flux_params)

T_bcs = TracerBoundaryConditions(grid,
    bottom = ValueBoundaryCondition(T_min),
       top = temperature_flux_bc
)

## Turbulent diffusivity closure

closure = AnisotropicDiffusivity(νh=500, νz=1e-2, κh=100, κz=1e-2)

## Model setup

@info "Model setup..."

model = IncompressibleModel(
           architecture = CPU(),
                   grid = grid,
            timestepper = :RungeKutta3,
              advection = WENO5(),
               coriolis = BetaPlane(latitude=45),
               buoyancy = SeawaterBuoyancy(constant_salinity=true),
                tracers = :T,
                closure = closure,
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs)
)

## Initial condition

@info "Setting initial conditions..."

# a stable density gradient with random noise superposed.
T₀(x, y, z) = temperature_flux_params.T_min + temperature_flux_params.ΔT/2 * (1 + z / grid.Lz)
set!(model, T=T₀)

# set!(model, T=temperature_flux_params.T_mid)

## Simulation setup

@info "Setting up simulation..."

u_max = FieldMaximum(abs, model.velocities.u)
v_max = FieldMaximum(abs, model.velocities.v)
w_max = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    K = 10
    convective_adjustment!(model, get_Δt(simulation), K)

    T_interior = interior(model.tracers.T)
    T_min, T_max = extrema(T_interior)
    T_mean = mean(T_interior)

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, u_max = (%.1e, %.1e, %.1e) m/s, T: (min=%.2f, mean=%.2f, max=%.2f), wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   u_max(), v_max(), w_max(), T_min, T_mean, T_max,
                   prettytime(1e-9 * (time_ns() - wall_clock)))

    @info msg

    return nothing
end

wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.5, Δt=1hour, max_change=1.1, max_Δt=1hour)

simulation = Simulation(model, Δt=wizard, stop_time=2years, iteration_interval=1, progress=print_progress)

## Set up output writers

@info "Setting up output writers..."

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.velocities, model.tracers),
                       schedule=TimeInterval(1day), filepath="double_gyre.nc", mode="c")

## Running the simulation

@info "Running simulation..."
run!(simulation)
