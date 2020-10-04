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
Q  = 75
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
               enforce_fluxes,
               standardize⁻¹_wT,
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
end

mp4(anim, "oceananigans_free_convection_$(Q)W.mp4", fps=60)

close(ds)
