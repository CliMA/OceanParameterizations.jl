using Printf
using LinearAlgebra

using NCDatasets
using BSON
using Plots

using Flux: relu
using DiffEqFlux: FastChain, FastDense

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.BoundaryConditions
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
##### Free convection model setup
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

#####
##### Neural PDE forcing function
#####

standardization = BSON.load("standardization.bson")[:standardization]
p_NN = BSON.load("free_convection_neural_pde_parameters.bson")[:best_weights]

μ_T, σ_T = standardization.T.μ, standardization.T.σ
μ_wT, σ_wT = standardization.wT.μ, standardization.wT.σ

standardize_T(x) = (x - μ_T) / σ_T
standardize⁻¹_T(y) = σ_T * y + μ_T
standardize_wT(x) = (x - μ_wT) / σ_wT
standardize⁻¹_wT(y) = σ_wT * y + μ_wT

standardize_T(T, p) = standardize_T.(T)
enforce_fluxes(wT, p) = cat(0, wT, Q / (ρ₀ * cₚ), dims=1)
standardize⁻¹_wT(wT, p) = standardize⁻¹_wT.(wT)

function ∂z_wT(wT, p)
    wT_field = ZFaceField(CPU(), grid)
    set!(wT_field, reshape(wT, (1, 1, Nz+1)))
    fill_halo_regions!(wT_field, CPU(), nothing, nothing)
    ∂z_wT_field = ComputedField(@at (Cell, Cell, Cell) ∂z(wT_field))
    compute!(∂z_wT_field)
    return interior(∂z_wT_field)[:]
end

Nz = grid.Nz
NN = FastChain(standardize_T,
               FastDense( Nz, 4Nz, relu),
               FastDense(4Nz, 4Nz, relu),
               FastDense(4Nz, Nz-1),
               standardize⁻¹_wT,
               enforce_fluxes,
               ∂z_wT)

∂z_wT_NN = zeros(Nz)
forcing_params = (∂z_wT_NN=∂z_wT_NN,)
@inline neural_network_∂z_wT(i, j, k, grid, clock, model_fields, p) = - p.∂z_wT_NN[k]
T_forcing = Forcing(neural_network_∂z_wT, discrete_form=true, parameters=forcing_params)

#####
##### Set up and run free convection
#####

model_convective_adjustment = IncompressibleModel(grid=grid, boundary_conditions=(T=T_bcs,))
model_neural_network = IncompressibleModel(grid=grid, boundary_conditions=(T=T_bcs,), forcing=(T=T_forcing,))

T₀(x, y, z) = 20 + 0.01z
set!(model_convective_adjustment, T=T₀)
set!(model_neural_network, T=T₀)

Δt = 10minute
stop_time = 6day
Nt = Int(stop_time / Δt)

ds = ds = NCDataset("free_convection_horizontal_averages_$(Q)W.nc")

T_NN_fig = zeros(Nz, Nt+1)
T_NN_fig[:, 1] .= interior(model_neural_network.tracers.T)[:]

anim = @animate for n in 1:Nt
    T = interior(model_neural_network.tracers.T)[:]
    ∂z_wT_NN .= NN(T, p_NN)

    time_step!(model_convective_adjustment, Δt)
    time_step!(model_neural_network, Δt)

    convective_adjustment!(model_convective_adjustment, Δt, K)
    convective_adjustment!(model_neural_network, Δt, K)

    time_str = @sprintf("%.2f days", model_neural_network.clock.time / day)
    @info "t = $time_str"

    T_CA = interior(model_convective_adjustment.tracers.T)[:]
    T_NN = interior(model_neural_network.tracers.T)[:]
    z = znodes(Cell, grid)
    
    plot(ds["T"][:, n], ds["zC"][:], linewidth=2, xlim=(19, 20), ylim=(-100, 0),
         label="Oceananigans 3D", xlabel="Temperature (°C)", ylabel="Depth z (meters)",
         title="Free convection: $time_str", legend=:bottomright, show=false)

    # plot!(T_CA, z, linewidth=2, label="Oceananigans 1D + convective adjustment", show=false)

    plot!(T_NN, z, linewidth=2, label="Oceananigans 1D + convective adjustment + neural PDE", show=false)

    T_NN_fig[:, n+1] .= T_NN
end

mp4(anim, "oceananigans_free_convection_$(Q)W.mp4", fps=60)

import PyPlot
const plt = PyPlot
const Line2D = plt.matplotlib.lines.Line2D
const Patch = plt.matplotlib.patches.Patch

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)

time_indices = (1, 109, 433, 865)
colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")

for (n, color) in zip(time_indices, colors)
    ax.plot(ds["T"][:, n], ds["zC"][:], color=color, alpha=0.4)
    ax.plot(T_NN_fig[:, n], znodes(Cell, grid), color=color, linestyle="--")
end

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("z (m)")
ax.set_xlim([19, 20])
ax.set_ylim([-100, 0])

ax.text(19.98, 1, "0 hours",  color="tab:blue",   rotation=45)
ax.text(19.79, 1, "18 hours", color="tab:orange", rotation=45)
ax.text(19.60, 1, "3 days",   color="tab:green",  rotation=45)
ax.text(19.46, 1, "6 days",   color="tab:red",    rotation=45)

custom_lines = [
    Line2D([0], [0], color="black", linestyle="-", alpha=0.5),
    Line2D([0], [0], color="black", linestyle="--")
]

ax.legend(custom_lines, ["Oceananigans 3D", "Oceananigans 1D + neural PDE"], loc="lower right", frameon=false)

plt.savefig("neurips_npde_figure.png")

close(ds)

grid_points = 32
T, wT, z = ds["T"], ds["wT"], ds["zC"]
zF_coarse = coarse_grain(ds["zF"], grid_points+1, Face)

S_T, S⁻¹_wT = standardization.T.standardize, standardization.wT.standardize⁻¹

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)

time_indices = (109, 433, 865) |> reverse
colors = ("tab:orange", "tab:green", "tab:red") |> reverse

for (n, color) in zip(time_indices, colors)
    wT = ds["wT"][:, n]
    wT[end] = Q / ρ₀ / cₚ
    ax.plot(wT / 1e-5, ds["zF"][:], color=color, alpha=0.5)

    wT_NN = coarse_grain(T[:, n], grid_points, Cell) .|> S_T |> NN .|> S⁻¹_wT
    wT_NN[end-1] = (wT_NN[end-2] + wT_NN[end]) / 2
    ax.plot(wT_NN / 1e-5, zF_coarse, color=color, linestyle="--")
end

ax.set_xlabel("Turbulent vertical heat flux (×10⁻⁵ m/s K)")
ax.set_ylabel("z (m)")
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-100, 0])

custom_lines = [
    Patch(facecolor="tab:orange"),
    Patch(facecolor="tab:green"),
    Patch(facecolor="tab:red"),
    Line2D([0], [0], color="black", linestyle="-", alpha=0.5),
    Line2D([0], [0], color="black", linestyle="--")
]

ax.legend(custom_lines, ["18 hours", "3 days", "6 days", "Oceananigans 3D", "Neural network"], loc="lower right", frameon=false)

plt.savefig("neurips_heat_flux_figure.png")
