module FreeConvection

export
    zC, zF,
    FreeConvectionTrainingDataInput, rescale,
    FreeConvectionNDE, ConvectiveAdjustmentNDE, FreeConvectionNDEParameters, initial_condition,
    solve_free_convection_nde, solve_convective_adjustment_nde, free_convection_solution,
    animate_variable, convection_training_data, animate_learned_heat_flux

using Logging
using Printf
using Statistics
using DataDeps
using DimensionalData
using GeoData
using NCDatasets
using Plots
using Oceananigans.Utils
using OceanParameterizations

using GeoData: GeoXDim, GeoYDim, GeoZDim
using Oceananigans: OceananigansLogger, Cell, Face

@dim zC ZDim "z"
@dim zF ZDim "z"

# Should not have saved constant units as strings...
nc_constant(attr) = parse(Float64, attr |> split |> first)

location_z(nc_var) = "zC" in dimnames(nc_var) ? Cell : Face

include("animations.jl")

"""
    FreeConvectionTrainingDataInput{Θ, B, T}

A container for holding free convection training data inputs.
"""
struct FreeConvectionTrainingDataInput{Θ, B, T}
    temperature :: Θ
    bottom_flux :: B
       top_flux :: T
end

rescale(old, T_scaling, wT_scaling) =
    FreeConvectionTrainingDataInput(T_scaling.(old.temperature), wT_scaling(old.bottom_flux), wT_scaling(old.top_flux))

"""
    training_data(ϕ; grid_points)

Return a `Nt × grid_points` array of coarse-grained data from the NetCDF variable `ϕ` where `Nt` is the number of times.
"""
function convection_training_data(ϕ; grid_points, iterations=nothing, scaling=identity)
    Nz, Nt = size(ϕ)
    loc = location_z(ϕ)

    iterations = isnothing(iterations) ? (1:Nt) : iterations
    data = cat((coarse_grain(ϕ[:, n], grid_points, loc) for n in iterations)..., dims=2)

    return scaling.(data)
end

function FreeConvectionNDE(NN, ds; grid_points, iterations=nothing)
    weights, reconstruct = Flux.destructure(NN)

    H = abs(ds["zF"][1]) # Domain height
    τ = ds["time"][end]  # Simulation length
    zC = coarse_grain(ds["zC"], grid_points, Cell)
    Δẑ = diff(zC)[1] / H  # Non-dimensional grid spacing
    Dzᶠ = Dᶠ(grid_points, Δẑ) # Differentiation matrix operator

    if isnothing(iterations)
        iterations = 1:length(ds["time"])
    end

    """
    Non-dimensional PDE is

        ∂T/∂t = - σ_wT/σ_T * τ/H * ∂/∂z(wT)
    """
    function ∂T∂t(T, p, t)
        weights = p[1:end-6]
        bottom_flux, top_flux, σ_T, σ_wT, H, τ = p[end-5:end]

        NN = reconstruct(weights)
        wT_interior = NN(T)
        wT = [bottom_flux; wT_interior; top_flux]
        ∂z_wT = Dzᶠ * σ_wT/σ_T * τ/H * wT
        return -∂z_wT
    end

    Nt = length(ds["time"])
    tspan = (0.0, maximum(iterations) / Nt)
    saveat = range(tspan[1], tspan[2], length = length(iterations))

    # We set the initial condition to `nothing`. We set it to some actual
    # initial condition when calling `solve`.
    return ODEProblem(∂T∂t, nothing, tspan, saveat=saveat)
end

function ConvectiveAdjustmentNDE(NN, ds; grid_points, iterations=nothing)
    weights, reconstruct = Flux.destructure(NN)

    H = abs(ds["zF"][1]) # Domain height
    τ = ds["time"][end]  # Simulation length
    zC = coarse_grain(ds["zC"], grid_points, Cell)
    Δẑ = diff(zC)[1] / H  # Non-dimensional grid spacing

    # Differentiation matrix operators
    Dzᶠ = Dᶠ(grid_points, Δẑ)
    Dzᶜ = Dᶜ(grid_points, Δẑ)

    if isnothing(iterations)
        iterations = 1:length(ds["time"])
    end

    """
    Non-dimensional PDE is

        ∂T/∂t = - σ_wT/σ_T * τ/H * ∂/∂z(wT + K ∂T/∂z)

    where K = 0 if ∂T/∂z < 0 and K = 100 if ∂T/∂z > 0.
    """
    function ∂T∂t(T, p, t)
        weights = p[1:end-6]
        bottom_flux, top_flux, σ_T, σ_wT, H, τ = p[end-5:end]

        # Turbulent heat flux
        NN = reconstruct(weights)
        wT_interior = NN(T)
        wT = [bottom_flux; wT_interior; top_flux]
        ∂z_wT = Dzᶠ * wT

        # Convective adjustment
        ∂T∂z = Dzᶜ * T
        ∂z_K∂T∂z = Dzᶠ * min.(0, 100 * ∂T∂z)

        return σ_wT/σ_T * τ/H * (- ∂z_wT .+ ∂z_K∂T∂z)
    end

    Nt = length(ds["time"])
    tspan = (0.0, maximum(iterations) / Nt)
    saveat = range(tspan[1], tspan[2], length = length(iterations))

    # We set the initial condition to `nothing`. We set it to some actual
    # initial condition when calling `solve`.
    return ODEProblem(∂T∂t, nothing, tspan, saveat=saveat)
end

function FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    H = abs(ds["zF"][1]) # Domain height
    τ = ds["time"][end]  # Simulation length

    Q  = nc_constant(ds.attrib["Heat flux"])
    ρ₀ = nc_constant(ds.attrib["Reference density"])
    cₚ = nc_constant(ds.attrib["Specific_heat_capacity"])

    bottom_flux = wT_scaling(0)
    top_flux = wT_scaling(Q / (ρ₀ * cₚ))

    fixed_params = [bottom_flux, top_flux, T_scaling.σ, wT_scaling.σ, H, τ]
end

function initial_condition(ds; grid_points, scaling=identity)
    T₀ = ds["T"][:, 1]
    T₀ = coarse_grain(T₀, grid_points, Cell)
    return scaling.(T₀)
end

function solve_free_convection_nde(nde, NN, T₀, alg, nde_params)
    nn_weights, _ = Flux.destructure(NN)
    return solve(nde, alg, u0=T₀, p=[nn_weights; nde_params],
                 sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end

function solve_convective_adjustment_nde(nde, NN, T₀, alg, nde_params)
    nn_weights, _ = Flux.destructure(NN)
    return solve(nde, alg, reltol=1e-3, u0=T₀, p=[nn_weights; nde_params],
                 sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end

function animate_learned_heat_flux(ds, NN, T_scaling, wT_scaling; grid_points, filepath, frameskip=1, fps=15)
    T, wT, zF = ds["T"], ds["wT"], ds["zF"]
    Nz, Nt = size(T)
    zF_coarse = coarse_grain(zF, grid_points+1, Face)

    Q  = nc_constant(ds.attrib["Heat flux"])
    ρ₀ = nc_constant(ds.attrib["Reference density"])
    cₚ = nc_constant(ds.attrib["Specific_heat_capacity"])

    bottom_flux = 0.0 |> wT_scaling
    top_flux = Q / (ρ₀ * cₚ) |> wT_scaling

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $filepath [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / days)

        wT_fine = wT[:, n]
        wT_fine[Nz+1] = unscale(top_flux, wT_scaling)

        plot(wT_fine, zF, linewidth=2, xlim=(-1e-5, 3e-5), ylim=(-100, 0),
             label="Oceananigans wT", xlabel="Heat flux", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        temperature = T_scaling.(coarse_grain(T[:, n], grid_points, Cell))
        input = FreeConvectionTrainingDataInput(temperature, bottom_flux, top_flux)
        wT_NN = unscale.(NN(input), Ref(wT_scaling))
        # wT_NN = inv(wT_scaling)(NN(input))

        plot!(wT_NN, zF_coarse, linewidth=2, label="Neural network")
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return nothing
end

const ENGAGING_LESBRARY_DIR = "https://engaging-web.mit.edu/~alir/lesbrary"

const LESBRARY_DATA_DEPS = (
    DataDep("lesbrary_free_convection_1",
            "proto-LESbrary.jl free convection statistics (Qb = 5×10⁻⁷ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb5.0e-07_f1.0e-04_Nh256_Nz128_pilot2", "statistics.nc")),
    DataDep("lesbrary_free_convection_2",
            "proto-LESbrary.jl free convection statistics (Qb = 2.5×10⁻⁷ m²/s³)",
            joinpath(ENGAGING_LESBRARY_DIR, "three_layer_constant_fluxes_cubic_hr48_Qu0.0e+00_Qb2.5e-07_f1.0e-04_Nh256_Nz128_pilot3", "statistics.nc")),
)

function __init__()
    Logging.global_logger(OceananigansLogger())
    return nothing
end

end # module
