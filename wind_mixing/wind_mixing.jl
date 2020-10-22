using Printf

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations
using Oceananigans.Utils

# Some physical constants
const f₀ = 1e-4  # Coriolis parameter [s⁻¹]
const ρ₀ = 1027  # Density of seawater [kg/m³]
const cₚ = 4000  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Set up grid
N, L = 128, 100
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(N, N, N), x=(0, L), y=(0, L), z=(-L, 0))

# Set up wind stress
τ = 0.02  # Wind stress [N/m^2]
u_top_bc = FluxBoundaryCondition(τ / ρ₀)

Q = 0  # Upward surface heat flux [W/m²]
T_top_bc = FluxBoundaryCondition(Q / (ρ₀ * cₚ))

∂T∂z = 0.01  # Initial temperature stratification [°C/m]
T_bot_bc = GradientBoundaryCondition(∂T∂z)

u_bcs = UVelocityBoundaryConditions(grid, top=u_top_bc)
T_bcs = TracerBoundaryConditions(grid, top=T_top_bc, bottom=T_bot_bc)

# Set up model
model = IncompressibleModel(
            architecture = GPU(),
                    grid = grid,
             timestepper = :RungeKutta3,
               advection = WENO5(),
                coriolis = FPlane(f=f₀),
                 tracers = :T,
                buoyancy = SeawaterBuoyancy(constant_salinity=true),
                 closure = AnisotropicMinimumDissipation(),
     boundary_conditions = (u=u_bcs, T=T_bcs)
)

# Set initial conditions
ε(σ) = σ * randn() # Gaussian noise with mean 0 and variance μ
T₀(x, y, z) = 20 + ∂T∂z * z + ε(1e-10) * exp(4z/L)
set!(model, T=T₀)

# Set up adaptive time stepping
wizard = TimeStepWizard(cfl=0.3, Δt=3.0, max_change=1.2, max_Δt=30.0)
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

    @info @sprintf("[%05.2f%%] i: %d, t: %s, umax: (%6.3g, %6.3g, %6.3g) m/s, T: (min=%6.4f, max=%6.4f), CFL: %6.4g, next Δt: %s",
                   progress, i, prettytime(t), u_max, v_max, w_max, T_min, T_max, cfl(model), prettytime(simulation.Δt.Δt))
end

# Set up simulation
simulation = Simulation(model, Δt=wizard, stop_time=8days, progress=print_simulation_stats,
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

(u, v, w), T = model.velocities, model.tracers.T

L⁻²∫u_dxdy = AveragedField(u, dims=(1, 2))
L⁻²∫v_dxdy = AveragedField(v, dims=(1, 2))
L⁻²∫T_dxdy = AveragedField(T, dims=(1, 2))

L⁻²∫uu_dxdy = AveragedField(u * u, dims=(1, 2))
L⁻²∫vv_dxdy = AveragedField(v * v, dims=(1, 2))
L⁻²∫ww_dxdy = AveragedField(w * w, dims=(1, 2))
L⁻²∫uv_dxdy = AveragedField(u * v, dims=(1, 2))
L⁻²∫uw_dxdy = AveragedField(u * w, dims=(1, 2))
L⁻²∫vw_dxdy = AveragedField(v * w, dims=(1, 2))

L⁻²∫uT_dxdy = AveragedField(u * T, dims=(1, 2))
L⁻²∫vT_dxdy = AveragedField(v * T, dims=(1, 2))
L⁻²∫wT_dxdy = AveragedField(w * T, dims=(1, 2))

profiles = Dict(
    "u"  => L⁻²∫u_dxdy,
    "v"  => L⁻²∫v_dxdy,
    "T"  => L⁻²∫T_dxdy,
    "uu" => L⁻²∫uu_dxdy,
    "vv" => L⁻²∫vv_dxdy,
    "ww" => L⁻²∫ww_dxdy,
    "uv" => L⁻²∫uv_dxdy,
    "uw" => L⁻²∫uw_dxdy,
    "vw" => L⁻²∫vw_dxdy,
    "uT" => L⁻²∫uT_dxdy,
    "vT" => L⁻²∫vT_dxdy,
    "wT" => L⁻²∫wT_dxdy
)

nc_filepath = "wind_mixing_horizontal_averages_$(τ)Nm2.nc"
simulation.output_writers[:profiles] =
    NetCDFOutputWriter(model, profiles, filepath=nc_filepath,
                       global_attributes=global_attributes,
                       schedule=TimeInterval(10minutes), verbose=true)

run!(simulation)
