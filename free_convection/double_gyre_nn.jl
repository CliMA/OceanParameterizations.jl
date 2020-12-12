# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.BoundaryConditions
using Oceananigans.Advection
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

using Oceananigans.Fields: PressureField

using Printf
using BSON
using ClimateParameterizations
using Flux
using LinearAlgebra

#####
##### Convective adjustment
#####

function convective_adjustment!(model, Δt, K)
    grid = model.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δz = model.grid.Δz
    b = model.tracers.b

    ∂b∂z = ComputedField(@at (Cell, Cell, Cell) ∂z(b))
    compute!(∂b∂z)

    κ = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        κ[i, j, k] = ∂b∂z[i, j, k] < 0 ? K : 0
    end

    b_interior = interior(b)
    b′ = zeros(Nx, Ny, Nz)

    for j in 1:Ny, i in 1:Nx
        ld = [-Δt/Δz^2 * κ[i, j, k]   for k in 2:Nz]
        ud = [-Δt/Δz^2 * κ[i, j, k+1] for k in 1:Nz-1]

        d = zeros(Nz)
        for k in 1:Nz-1
            d[k] = 1 + Δt/Δz^2 * (κ[i, j, k] + κ[i, j, k+1])
        end
        d[Nz] = 1 + Δt/Δz^2 * κ[i, j, Nz]

        𝓛 = Tridiagonal(ld, d, ud)

        b′[i, j, :] .= 𝓛 \ b_interior[i, j, :]
    end

    set!(model, b=b′)

    return nothing
end

grid = RegularCartesianGrid(size=(64, 64, 32), x=(-2e5, 2e5), y=(-3e5, 3e5), z=(-1e3, 0),
                            topology=(Bounded, Bounded, Bounded))

# ## Boundary conditions
#
# ### Wind stress

@inline wind_stress(x, y, t, p) = - p.τ * cos(2π * y / p.Ly)

surface_stress_u_bc = BoundaryCondition(Oceananigans.Flux, wind_stress, parameters=(τ=1e-4, Ly=grid.Ly))

# ### Bottom drag

@inline bottom_drag_u(x, y, t, u, p) = - p.μ * p.Lz * u
@inline bottom_drag_v(x, y, t, v, p) = - p.μ * p.Lz * v

bottom_drag_u_bc = BoundaryCondition(Oceananigans.Flux, bottom_drag_u, field_dependencies=:u, parameters=(μ=1/180day, Lz=grid.Lz))
bottom_drag_v_bc = BoundaryCondition(Oceananigans.Flux, bottom_drag_v, field_dependencies=:v, parameters=(μ=1/180day, Lz=grid.Lz))

u_bcs = UVelocityBoundaryConditions(grid, top = surface_stress_u_bc, bottom = bottom_drag_u_bc)
v_bcs = VVelocityBoundaryConditions(grid, bottom = bottom_drag_v_bc)

# ### Buoyancy relaxation

@inline buoyancy_flux(x, y, t, b, p) = - p.μ * (b - p.Δb / p.Ly * y)

buoyancy_flux_params = (μ=1/day, Δb=0.05, Ly=grid.Ly)
buoyancy_flux_bc = BoundaryCondition(Oceananigans.Flux, buoyancy_flux,
                                     field_dependencies = :b)

b_bcs = TracerBoundaryConditions(grid, # top = buoyancy_flux_bc,
                                       bottom = BoundaryCondition(Value, 0))

# NN forcing

neural_network_parameters = BSON.load("convective_adjustment_nde.bson")

NN = neural_network_parameters[:neural_network]
T_scaling = neural_network_parameters[:T_scaling]
wT_scaling = neural_network_parameters[:wT_scaling]

enforce_fluxes(wb, bottom_flux, top_flux) = cat(bottom_flux, wb, top_flux, dims=1)

function ∂z_wb(wb)
    wb_field = ZFaceField(CPU(), grid)
    set!(wb_field, wb)
    fill_halo_regions!(wb_field, CPU(), nothing, nothing)
    ∂z_wb_field = ComputedField(@at (Cell, Cell, Cell) ∂z(wb_field))
    compute!(∂z_wb_field)
    return interior(∂z_wb_field)
end

Nx, Ny, Nz = size(grid)
∂z_wb_NN = zeros(Nx, Ny, Nz)
b_neural_network_params = (∂z_wb_NN=∂z_wb_NN,)
@inline neural_network_∂z_wb(i, j, k, grid, clock, model_fields, p) = - p.∂z_wb_NN[i, j, k]
b_forcing = Forcing(neural_network_∂z_wb, discrete_form=true, parameters=b_neural_network_params)

# ## Turbulence closure
closure = AnisotropicDiffusivity(νh=500, νz=1e-2, κh=100, κz=1e-2)

# ## Model building

model = IncompressibleModel(architecture = CPU(),
                            timestepper = :RungeKutta3, 
                            advection = UpwindBiasedFifthOrder(),
                            grid = grid,
                            coriolis = BetaPlane(latitude=45),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            closure = closure,
                            boundary_conditions = (u=u_bcs, v=v_bcs, b=b_bcs),
			    forcing = (b=b_forcing,))

# ## Initial conditions

bᵢ(x, y, z) = buoyancy_flux_params.Δb * (1 + z / grid.Lz)

set!(model, b=bᵢ)

# ## Simulation setup

max_Δt = 1 / 10model.coriolis.f₀

wizard = TimeStepWizard(cfl=1.0, Δt=hour/2, max_change=1.1, max_Δt=max_Δt)

# Finally, we set up and run the the simulation.

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

α = 1.67e-4
g = 9.8

function print_progress(simulation)
    model = simulation.model

    convective_adjustment!(model, simulation.Δt.Δt, 1)

    b = interior(model.tracers.b)
    wb = zeros(Nx, Ny, Nz+1)
    for i in 1:Nx, j in 1:Ny
	surface_flux_ij = buoyancy_flux(grid.xC[i], grid.yC[j], model.clock.time, b[i, j, Nz], buoyancy_flux_params)

	b_profile = b[i, j, :]
	T_profile = α * g * b_profile
	T_profile_scaled = @. 19 + T_profile/20
	wT_interior_ij = NN(T_scaling.(T_profile_scaled))
	wb_interior_ij = inv(wT_scaling).(wT_interior_ij) / (α * g)
        wb_ij = enforce_fluxes(wb_interior_ij, 0, surface_flux_ij)
        wb[i, j, :] .= wb_ij
    end
    ∂z_wb_NN .= ∂z_wb(wb)

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

simulation = Simulation(model, Δt=wizard, stop_time=365days, iteration_interval=10, progress=print_progress)

# ## Output

u, v, w = model.velocities
b = model.tracers.b

speed = ComputedField(u^2 + v^2)
buoyancy_variance = ComputedField(b^2)

outputs = merge(model.velocities, model.tracers, (speed=speed, b²=buoyancy_variance))

simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs,
                                                        schedule = TimeInterval(2days),
                                                        filepath = "double_gyre_nn.nc")

p = PressureField(model)
barotropic_p = AveragedField(p, dims=3)
barotropic_u = AveragedField(model.velocities.u, dims=3)
barotropic_v = AveragedField(model.velocities.v, dims=3)

simulation.output_writers[:barotropic_velocities] =
    NetCDFOutputWriter(model, (u=barotropic_u, v=barotropic_v, p=barotropic_p),
                       schedule = AveragedTimeInterval(30days, window=10days),
                       filepath = "double_gyre_circulation_nn.nc")

run!(simulation)

