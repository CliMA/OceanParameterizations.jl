# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Printf
using Statistics

using Oceananigans
using Oceananigans.Fields
using Oceananigans.Advection
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.Utils

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

@inline b_reference(y, p) = p.Δb / p.Ly * y
@inline buoyancy_flux(x, y, t, b, p) = @inbounds - p.μ * (b - b_reference(y, p))

buoyancy_flux_params = (μ=1/day, Δb=0.06, Ly=grid.Ly)
buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, field_dependencies=:b, parameters=buoyancy_flux_params)

b_bcs = TracerBoundaryConditions(grid,
    bottom = ValueBoundaryCondition(0),
       top = buoyancy_flux_bc
)

## Turbulent diffusivity closure

closure = AnisotropicDiffusivity(νh=500, νz=1e-2, κh=100, κz=1e-2)

## Model setup

@info "Model setup..."

model = IncompressibleModel(
           architecture = CPU(),
            timestepper = :RungeKutta3,
              advection = WENO5(),
                   grid = grid,
               coriolis = BetaPlane(latitude=45),
               buoyancy = BuoyancyTracer(),
                tracers = :b,
                closure = closure,
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs)
)

## Initial condition

@info "Setting initial conditions..."

# a stable density gradient with random noise superposed.
b₀(x, y, z) = buoyancy_flux_params.Δb * (1 + z / grid.Lz)

set!(model, b=b₀)

## Simulation setup

@info "Setting up simulation..."

wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.5, Δt=1hour, max_change=1.1, max_Δt=1hour)

u_max = FieldMaximum(abs, model.velocities.u)
v_max = FieldMaximum(abs, model.velocities.v)
w_max = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    b_interior = interior(model.tracers.b)
    b_min, b_max = extrema(b_interior)
    b_mean = mean(b_interior)

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, u_max = (%.1e, %.1e, %.1e) m/s, b: (min=%.1e, mean=%.1e, max=%.1e), wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   u_max(), v_max(), w_max(), b_min, b_mean, b_max,
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=20years, iteration_interval=1, progress=print_progress)

## Set up output writers

@info "Setting up output writers..."

u, v, w = model.velocities
speed = ComputedField(√(u^2 + v^2))

output_fields = merge(model.velocities, model.tracers, (speed=speed,))

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, output_fields, schedule=TimeInterval(1day), filepath="good_double_gyre.nc", mode="c")

## Running the simulation

@info "Running simulation..."
run!(simulation)
