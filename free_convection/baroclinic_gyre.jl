using Printf
using LinearAlgebra

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Advection
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations
using Oceananigans.Utils

const km = kilometer

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

topo = (Bounded, Bounded, Bounded)
domain = (x=(-3000km, 3000km), y=(-3000km, 3000km), z=(-1.8km, 0))
grid = RegularCartesianGrid(topology=topo, size = (60, 60, 32), halo = (3, 3, 3); domain...)

no_slip = BoundaryCondition(Value, 0)

u_bc_params = (τ=0.1, ρ₀=1027, Ly=grid.Ly, Δz=grid.Δz)
@inline wind_stress(x, y, t, p) = - p.τ / (p.ρ₀ * p.Δz) * cos(2π * y / p.Ly)

u_bc_top = BoundaryCondition(Flux, wind_stress, parameters=u_bc_params)
u_bcs = UVelocityBoundaryConditions(grid, top=u_bc_top, south=no_slip, north=no_slip)

v_bcs = VVelocityBoundaryConditions(grid, east=no_slip, west=no_slip)
w_bcs = WVelocityBoundaryConditions(grid, east=no_slip, west=no_slip, north=no_slip, south=no_slip)

T_bc_params = (τ_T = 30day, T_min=0, T_max=30, Ly=grid.Ly)
@inline surface_temperature(y, p) = (p.T_max - p.T_min) / p.Ly * y
@inline surface_temperature_relaxation(i, j, grid, clock, model_fields, p) =
    @inbounds -1/p.τ_T * (model_fields.T[i, j, grid.Nz] - surface_temperature(grid.yC[j], p))

T_bc_top = BoundaryCondition(Flux, surface_temperature_relaxation, discrete_form=true, parameters=T_bc_params)
T_bcs = TracerBoundaryConditions(grid, top=T_bc_top)

closure = AnisotropicDiffusivity(νh=5000, νz=1e-2, κh=1000, κz=1e-5)

model = IncompressibleModel(
                   grid = grid,
           architecture = CPU(),
            timestepper = :RungeKutta3,
              advection = UpwindBiasedThirdOrder(),
               coriolis = BetaPlane(latitude=45),
                tracers = :T,
               buoyancy = SeawaterBuoyancy(constant_salinity=true),
                closure = closure,
    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs)
)

T_bottom, T_top = 2, 30
T₀(x, y, z) =  T_top + (T_top - T_bottom) * z / grid.Lz
set!(model, T=T₀)

fields = Dict("u" => model.velocities.u, "v" => model.velocities.v, "w" => model.velocities.w, "T" => model.tracers.T)
field_writer = NetCDFOutputWriter(model, fields, filename="baroclinic_gyre.nc", time_interval=1day)
                                              
max_Δt = min(0.1grid.Δz^2 / closure.κz, 0.1grid.Δx^2 / closure.νx)
wizard = TimeStepWizard(cfl=0.5, Δt=1minute, max_change = 1.1, max_Δt=max_Δt)

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

advective_cfl = AdvectiveCFL(wizard)
diffusive_cfl = DiffusiveCFL(wizard)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    convective_adjustment!(model, simulation.Δt.Δt, 100)

    @info @sprintf("i: %04d, t: %s, Δt: %s, U_max = (%.4e, %.4e, %.4e) m/s, CFL: (advective=%.4e, diffusive=%.4e), wall time: %s\n",
                   model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
                   umax(), vmax(), wmax(), advective_cfl(model), diffusive_cfl(model),
                   prettytime(1e-9 * (time_ns() - wall_clock)))

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=3*365day, iteration_interval=1, progress=print_progress)
simulation.output_writers[:fields] = field_writer

run!(simulation)


# # # Making a neat movie
# #
# # We look at the results by plotting vertical slices of $u$ and $w$, and a horizontal
# # slice of $w$ to look for Langmuir cells.

# # Making the coordinate arrays takes a few lines of code,

# x, y, z = nodes(model.tracers.b)
# x, y, z = x[:], y[:], z[:]
# nothing # hide

# # Next, we open the JLD2 file, and extract the iterations we ended up saving at,

# using JLD2, Plots

# file = jldopen(simulation.output_writers[:fields].filepath)

# iterations = parse.(Int, keys(file["timeseries/t"]))
# nothing # hide

# # This utility is handy for calculating nice contour intervals:

# function nice_divergent_levels(c, clim)
#     levels = range(-clim, stop=clim, length=20)

#     cmax = maximum(abs, c)

#     if clim < cmax # add levels on either end
#         levels = vcat([-cmax], range(-clim, stop=clim, length=10), [cmax])
#     end

#     return levels
# end
# nothing # hide

# # Finally, we're ready to animate.

# @info "Making an animation from the saved data..."

# anim = @animate for (i, iter) in enumerate(iterations)
    
#     @info "Drawing frame $i from iteration $iter \n"

#     ## Load 3D fields from file, omitting halo regions
#     u = file["timeseries/u/$iter"]
#     v = file["timeseries/v/$iter"]
#     w = file["timeseries/w/$iter"]
#     t = file["timeseries/t/$iter"]

#     ## Extract slices
#     uxy = 1/2 * (u[1:end-1, :, end] .+ u[2:end, :, end])
#     vxy = 1/2 * (v[:, 1:end-1, end] .+ v[:, 2:end, end])
#     wxy = w[:, :, 1]
    
#     speed = @. sqrt(uxy^2 + vxy^2)
    
#     ulim = 1.0
#     ulevels = nice_divergent_levels(u, ulim)

#     uxy_plot = heatmap(x / 1e3, y / 1e3, uxy';
#                               color = :balance,
#                         aspectratio = :equal,
#                               # clims = (-2, 2),
#                              # levels = ulevels,
#                               xlims = (-grid.Lx/2e3, grid.Lx/2e3),
#                               ylims = (-grid.Ly/2e3, grid.Ly/2e3),
#                              xlabel = "x (km)",
#                              ylabel = "y (km)")
                        
#      wxy_plot = heatmap(x / 1e3, y / 1e3, wxy';
#                                color = :balance,
#                          aspectratio = :equal,
#                                # clims = (-1e-2, 1e-2),
#                               # levels = ulevels,
#                                xlims = (-grid.Lx/2e3, grid.Lx/2e3),
#                                ylims = (-grid.Ly/2e3, grid.Ly/2e3),
#                               xlabel = "x (km)",
#                               ylabel = "y (km)")
                         
#     speed_plot = heatmap(x / 1e3, y / 1e3 , speed';
#                               color = :deep,
#                         aspectratio = :equal,
#                               clims = (0, 2.0),
#                              # levels = ulevels,
#                               xlims = (-grid.Lx/2e3, grid.Lx/2e3),
#                               ylims = (-grid.Ly/2e3, grid.Ly/2e3),
#                              xlabel = "x (km)",
#                              ylabel = "y (km)")
                             
#     plot(uxy_plot, speed_plot, size=(1100, 500), title = ["u(t="*string(round(t/day, digits=1))*" day)" "speed"])

#     iter == iterations[end] && close(file)
# end

# gif(anim, "double_gyre.gif", fps = 12) # hide
