# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Utils: minute, day

grid = RegularCartesianGrid(size=(96, 96, 32), x=(-2e6, 2e6), y=(-3e6, 3e6), z=(-2e3, 0),
                            topology=(Bounded, Bounded, Bounded))

# ## Boundary conditions
@info "bcs"
using Oceananigans.BoundaryConditions

@inline wind_stress(x, y, t, parameters) = - parameters.τ * cos(2π * y / parameters.L)

@inline u_bottom_stress(i, j, grid, clock, model_fields, parameters) =
    @inbounds - parameters.μ * parameters.H * model_fields.u[i, j, 1]

@inline v_bottom_stress(i, j, grid, clock, model_fields, parameters) =
    @inbounds - parameters.μ * parameters.H * model_fields.v[i, j, 1]

wind_stress_bc = BoundaryCondition(Flux, wind_stress, parameters = (τ=1e-4, L=grid.Ly))

u_bottom_stress_bc = BoundaryCondition(Flux, u_bottom_stress,
                                       discrete_form=true, parameters=(μ=1/30day, H=grid.Lz))

v_bottom_stress_bc = BoundaryCondition(Flux, v_bottom_stress,
                                       discrete_form=true, parameters=(μ=1/30day, H=grid.Lz))

u_bcs = UVelocityBoundaryConditions(grid,
                                     north = BoundaryCondition(Value, 0),
                                     south = BoundaryCondition(Value, 0),
                                       top = wind_stress_bc,
                                    bottom = u_bottom_stress_bc)

v_bcs = VVelocityBoundaryConditions(grid,
                                      east = BoundaryCondition(Value, 0),
                                      west = BoundaryCondition(Value, 0),
                                    bottom = v_bottom_stress_bc)

w_bcs = WVelocityBoundaryConditions(grid,
                                    north = BoundaryCondition(Value, 0),
                                    south = BoundaryCondition(Value, 0),
                                     east = BoundaryCondition(Value, 0),
                                     west = BoundaryCondition(Value, 0))

b_reference(y, parameters) = parameters.Δb / parameters.Ly * y

using Oceananigans.Utils

@inline buoyancy_flux(i, j, grid, clock, model_fields, parameters) =
    @inbounds - parameters.μ * (model_fields.b[i, j, grid.Nz] - b_reference(grid.yC[j], parameters))

buoyancy_flux_bc = BoundaryCondition(Flux, buoyancy_flux,
                                     discrete_form = true,
                                     parameters = (μ=1/day, Δb=0.06, Ly=grid.Ly))

b_bcs = TracerBoundaryConditions(grid, 
                                 bottom = BoundaryCondition(Value, 0),
                                 top = buoyancy_flux_bc)

using Oceananigans, Oceananigans.TurbulenceClosures, Oceananigans.Advection

closure = AnisotropicDiffusivity(νh=500, νz=1e-2, κh=100, κz=1e-2)
@info "model"
model = IncompressibleModel(       architecture = CPU(),
                                    timestepper = :RungeKutta3, 
                                      advection = WENO5(),
                                           grid = grid,
                                       coriolis = BetaPlane(latitude=45),
                                       buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        closure = closure,
                            boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs))
nothing # hide

## Temperature initial condition: a stable density gradient with random noise superposed.
b₀(x, y, z) = b_bcs.top.condition.parameters.Δb * (1 + z / grid.Lz)

set!(model, b=b₀)

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

max_Δt = min(hour/2, 0.5 * min(grid.Δz^2 / closure.κz, grid.Δx^2 / closure.νx))

wizard = TimeStepWizard(cfl=0.5, Δt=hour/2, max_change=1.1, max_Δt=max_Δt)
nothing # hide

# Finally, we set up and run the the simulation.

using Oceananigans.Diagnostics, Printf

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   umax(), vmax(), wmax(),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end
@info "sim"
simulation = Simulation(model, Δt=wizard, stop_time=20years, iteration_interval=1, progress=print_progress)

# ## Set up output
#
# We set up an output writer that saves all velocity fields, tracer fields, and the subgrid
# turbulent diffusivity associated with `model.closure`. The `prefix` keyword argument
# to `JLD2OutputWriter` indicates that output will be saved in
# `double_gyre.jld2`.
@info "output"
using Oceananigans.OutputWriters

fields = Dict(
    "u" => model.velocities.u,
    "v" => model.velocities.v,
    "w" => model.velocities.w,
    "b" => model.tracers.b
)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, fields;
                                                        schedule=TimeInterval(2days),
                                                        filepath="double_gyre.nc")
@info "run"
run!(simulation)
