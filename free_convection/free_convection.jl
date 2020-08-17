using Printf
using Oceananigans
using Oceananigans.Utils
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

# Some physical constants
const f₀ = 1e-4  # Coriolis parameter [s⁻¹]
const ρ₀ = 1027  # Density of seawater [kg/m³]
const cₚ = 4000  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Set up grid
N, L = 256, 100
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(N, N, N), x=(0, L), y=(0, L), z=(-L, 0))

# Set up boundary conditions
Q = 100.0  # Upward surface heat flux [W/m²]
T_top_bc = FluxBoundaryCondition(Q / (ρ₀ * cₚ))

∂T∂z = 0.01  # Initial temperature stratification [°C/m]
T_bot_bc = GradientBoundaryCondition(∂T∂z)

T_bcs = TracerBoundaryConditions(grid, top=T_top_bc, bottom=T_bot_bc)

# Set up model
model = IncompressibleModel(
            architecture = GPU(),
                    grid = grid,
                coriolis = FPlane(f=f₀),
                 tracers = :T,
                buoyancy = SeawaterBuoyancy(constant_salinity=true),
                 closure = AnisotropicMinimumDissipation(),
     boundary_conditions = (T=T_bcs,)
)

# Set initial conditions
ε(σ) = σ * randn() # Gaussian noise with mean 0 and variance μ
T₀(x, y, z) = 20 + ∂T∂z * z + ε(1e-10) * exp(4z/L)
set!(model, T=T₀)

# Set up adaptive time stepping
wizard = TimeStepWizard(cfl=0.25, Δt=3.0, max_change=1.2, max_Δt=30.0)
cfl = AdvectiveCFL(wizard)

function print_simulation_stats(simulation)
    model = simulation.model
    i, t = model.clock.iteration, model.clock.time
    cfl = simulation.parameters.cfl

    progress = 100 * (t / simulation.stop_time)

    u_max = maximum(abs, model.velocities.u.data.parent)
    v_max = maximum(abs, model.velocities.v.data.parent)
    w_max = maximum(abs, model.velocities.w.data.parent)
    T_min = minimum(abs, model.tracers.T.data.parent)
    T_max = maximum(abs, model.tracers.T.data.parent)

    @info @sprintf("[%05.2f%%] i: %d, t: %.3f days, umax: (%6.3g, %6.3g, %6.3g) m/s, T: (min=%6.4f, max=%6.4f), CFL: %6.4g, next Δt: %.1f s",
                   progress, i, t / day, u_max, v_max, w_max, T_min, T_max, cfl(model), simulation.Δt.Δt)
end

# Set up simulation
simulation = Simulation(model, Δt=wizard, stop_time=8day, progress=print_simulation_stats,
                        iteration_interval=50, parameters=(cfl=cfl,))

#####
##### Set up output writers
#####

# Code credit: https://discourse.julialang.org/t/collecting-all-output-from-shell-commands/15592
function execute(cmd::Cmd)
    out, err = Pipe(), Pipe()

    process = run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
    close(out.in)
    close(err.in)

    return (stdout = out |> read |> String, stderr = err |> read |> String, code = process.exitcode)
end

global_attributes = Dict(
    "ClimateSurrogates.jl git commit SHA1" => execute(`git rev-parse HEAD`).stdout |> strip,
    "Reference density" => "$ρ₀ kg/m³",
    "Specific_heat_capacity" => "$cₚ J/kg/K",
    "Coriolis parameter" => "$f₀ s⁻¹",
    "Gravitational acceleration" => "$(model.buoyancy.gravitational_acceleration) m/s²",
    "Heat flux" => "$Q W/m²",
    "Initial stratification" => "$∂T∂z °C/m",
    "closure" => "$(model.closure)",
    "coriolis" => "$(model.coriolis)"
)


u, v, w = model.velocities
T = model.tracers.T

L⁻²∫u_dxdy = Average(u, dims=(1, 2), return_type=Array)
L⁻²∫v_dxdy = Average(v, dims=(1, 2), return_type=Array)
L⁻²∫T_dxdy = Average(T, dims=(1, 2), return_type=Array)

L⁻²∫uT_dxdy = Average(u * T, model, dims=(1, 2), return_type=Array)
L⁻²∫vT_dxdy = Average(v * T, model, dims=(1, 2), return_type=Array)
L⁻²∫wT_dxdy = Average(w * T, model, dims=(1, 2), return_type=Array)


profile_output_attributes = Dict(
    "u"  => Dict("longname" => "Horizontally averaged velocity in the x-direction", "units" => "m/s"),
    "v"  => Dict("longname" => "Horizontally averaged velocity in the y-direction", "units" => "m/s"),
    "T"  => Dict("longname" => "Horizontally averaged temperature", "units" => "°C"),
    "uT" => Dict("longname" => "Horizontally averaged heat flux in the x-direction", "units" => "°C m/s"),
    "vT" => Dict("longname" => "Horizontally averaged heat flux in the y-direction", "units" => "°C m/s"),
    "wT" => Dict("longname" => "Horizontally averaged heat flux in the z-direction", "units" => "°C m/s")
)


Hz = model.grid.Hz
profiles = Dict(
    "u"  => model -> L⁻²∫u_dxdy(model)[1+Hz:end-Hz],
    "v"  => model -> L⁻²∫v_dxdy(model)[1+Hz:end-Hz],
    "T"  => model -> L⁻²∫T_dxdy(model)[1+Hz:end-Hz],
    "uT" => model -> L⁻²∫uT_dxdy(model)[1+Hz:end-Hz],
    "vT" => model -> L⁻²∫vT_dxdy(model)[1+Hz:end-Hz],
    "wT" => model -> L⁻²∫wT_dxdy(model)[1+Hz:end-Hz]
)

profile_dims = Dict(k => ("zC",) for k in keys(profiles))

simulation.output_writers[:profiles] =
    NetCDFOutputWriter(model, profiles, filename="free_convection_horizontal_averages.nc",
                       global_attributes=global_attributes, output_attributes=profile_output_attributes,
                       dimensions=profile_dims, time_interval=10minute, verbose=true)

run!(simulation)

