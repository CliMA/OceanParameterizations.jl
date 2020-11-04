using Printf
using LinearAlgebra

using BSON

using Flux: relu
using DiffEqFlux: FastChain, FastDense

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.BoundaryConditions
using Oceananigans.Advection
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations
using Oceananigans.Utils

const km = kilometer

#####
##### Convective adjustment
#####

function convective_adjustment!(model, Δt, K)
    grid = model.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δz = model.grid.Δz
    T = model.tracers.T
    
    ∂T∂z = ComputedField(@at (Cell, Cell, Cell) ∂z(T))
    compute!(∂T∂z)

    κ = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        κ[i, j, k] = ∂T∂z[i, j, k] < 0 ? K : 0
    end

    T_interior = interior(T)
    T′ = zeros(Nx, Ny, Nz)

    for j in 1:Ny, i in 1:Nx
        ld = [-Δt/Δz^2 * κ[i, j, k]   for k in 2:Nz]
        ud = [-Δt/Δz^2 * κ[i, j, k+1] for k in 1:Nz-1]

        d = zeros(Nz)
        for k in 1:Nz-1
            d[k] = 1 + Δt/Δz^2 * (κ[i, j, k] + κ[i, j, k+1])
        end
        d[Nz] = 1 + Δt/Δz^2 * κ[i, j, Nz]

        𝓛 = Tridiagonal(ld, d, ud)
    
        T′[i, j, :] .= 𝓛 \ T_interior[i, j, :]
    end
    
    set!(model, T=T′)

    return nothing
end

#####
##### Baroclinic gyre setup
#####

topo = (Bounded, Bounded, Bounded)
domain = (x=(0, 6000km), y=(0, 6000km), z=(-1.8km, 0))
grid = RegularCartesianGrid(topology=topo, size=(60, 60, 32), halo=(3, 3, 3); domain...)

no_slip = BoundaryCondition(Value, 0)

u_bc_params = (τ=0.1, ρ₀=1027, Ly=grid.Ly, Δz=grid.Δz)
@inline wind_stress(x, y, t, p) = - p.τ / (p.ρ₀ * p.Δz) * cos(2π * y / p.Ly)

u_bc_top = BoundaryCondition(Flux, wind_stress, parameters=u_bc_params)
u_bcs = UVelocityBoundaryConditions(grid, top=u_bc_top, south=no_slip, north=no_slip)

v_bcs = VVelocityBoundaryConditions(grid, east=no_slip, west=no_slip)
w_bcs = WVelocityBoundaryConditions(grid, east=no_slip, west=no_slip, north=no_slip, south=no_slip)

#####
##### Neural PDE forcing function
#####

T_relaxation_params = (τ_T = 30day, T_min=0, T_max=30, Ly=grid.Ly)
@inline surface_temperature(y, p) = p.T_min + (p.T_max - p.T_min) / p.Ly * y
@inline surface_temperature_relaxation(i, j, k, grid, T, p) =
    @inbounds k == grid.Nz ? - 1/p.τ_T * (T[i, j, k] - surface_temperature(grid.yC[j], p)) : 0

standardization = BSON.load("standardization.bson")[:standardization]
p_NN = BSON.load("free_convection_neural_pde_parameters.bson")[:best_weights]

μ_T, σ_T = standardization.T.μ, standardization.T.σ
μ_wT, σ_wT = standardization.wT.μ, standardization.wT.σ

standardize_T(x) = (x - μ_T) / σ_T
standardize⁻¹_T(y) = σ_T * y + μ_T
standardize_wT(x) = (x - μ_wT) / σ_wT
standardize⁻¹_wT(y) = σ_wT * y + μ_wT

normalize_T(T, p) = @. 19 + T/20
standardize_T(T, p) = standardize_T.(T)
standardize⁻¹_wT(wT, p) = standardize⁻¹_wT.(wT)

enforce_fluxes(wT, bottom_flux, top_flux) = cat(bottom_flux, wT, top_flux, dims=1)

function ∂z_wT(wT)
    wT_field = ZFaceField(CPU(), grid)
    set!(wT_field, wT)
    fill_halo_regions!(wT_field, CPU(), nothing, nothing)
    ∂z_wT_field = ComputedField(@at (Cell, Cell, Cell) ∂z(wT_field))
    compute!(∂z_wT_field)
    return interior(∂z_wT_field)
end

Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

NN = FastChain(normalize_T,
               standardize_T,
               FastDense( Nz, 4Nz, relu),
               FastDense(4Nz, 4Nz, relu),
               FastDense(4Nz, Nz-1),
               standardize⁻¹_wT)

∂z_wT_NN = zeros(Nx, Ny, Nz)
T_neural_network_params = (∂z_wT_NN=∂z_wT_NN,)
@inline neural_network_∂z_wT(i, j, k, grid, clock, model_fields, p) = - p.∂z_wT_NN[i, j, k]
T_forcing = Forcing(neural_network_∂z_wT, discrete_form=true, parameters=T_neural_network_params)

#####
##### Continue setting up baroclinic gyre
#####

closure = AnisotropicDiffusivity(νh=5000, νz=1e-2, κh=1000, κz=1e-5)

model = IncompressibleModel(
                   grid = grid,
           architecture = CPU(),
            timestepper = :RungeKutta3,
              advection = UpwindBiasedThirdOrder(),
               coriolis = BetaPlane(latitude=15),
                tracers = :T,
               buoyancy = SeawaterBuoyancy(constant_salinity=true),
                closure = closure,
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                forcing = (T=T_forcing,)
)

T_bottom, T_top = 0, 30
T₀(x, y, z) =  T_top + (T_top - T_bottom) * z / grid.Lz
set!(model, T=T₀)

fields = Dict("u" => model.velocities.u, "v" => model.velocities.v, "w" => model.velocities.w, "T" => model.tracers.T)
field_writer = NetCDFOutputWriter(model, fields, filename="baroclinic_gyre_neural_network.nc", time_interval=1day)
                                              
max_Δt = min(0.1grid.Δz^2 / closure.κz, 0.1grid.Δx^2 / closure.νx)
wizard = TimeStepWizard(cfl=0.1, Δt=1minute, max_change = 1.1, max_Δt=max_Δt)

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

advective_cfl = AdvectiveCFL(wizard)
diffusive_cfl = DiffusiveCFL(wizard)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz

    convective_adjustment!(model, simulation.Δt.Δt, 1)

    T = interior(model.tracers.T)
    wT = zeros(Nx, Ny, Nz+1)
    for i in 1:Nx, j in 1:Ny
        surface_flux_ij = surface_temperature_relaxation(i, j, Nz, model.grid, model.tracers.T, T_relaxation_params)
        wT_ij = NN(T[i, j, :], p_NN)
        wT_ij = enforce_fluxes(wT_ij, 0, surface_flux_ij)
        wT[i, j, :] .= wT_ij
    end
    ∂z_wT_NN .= ∂z_wT(wT)

    @info @sprintf("i: %04d, t: %s, Δt: %s, U_max = (%.4e, %.4e, %.4e) m/s, CFL: (advective=%.4e, diffusive=%.4e), wall time: %s\n",
                   model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
                   umax(), vmax(), wmax(), advective_cfl(model), diffusive_cfl(model),
                   prettytime(1e-9 * (time_ns() - wall_clock)))

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=365day, iteration_interval=1, progress=print_progress)
simulation.output_writers[:fields] = field_writer

run!(simulation)
