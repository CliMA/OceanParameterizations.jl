module FreeConvection

export
    zC, zF,
    coarse_grain, add_surface_fluxes,
    animate_variable, animate_learned_heat_flux,
    FreeConvectionTrainingDataInput, rescale, wrangle_input_training_data, wrangle_output_training_data,
    FreeConvectionNDE, ConvectiveAdjustmentNDE, FreeConvectionNDEParameters, initial_condition,
    solve_free_convection_nde, solve_convective_adjustment_nde, free_convection_solution

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

using DimensionalData: basetypeof
using GeoData: AbstractGeoStack, window
using Oceananigans: OceananigansLogger, Cell, Face

@dim zC ZDim "z"
@dim zF ZDim "z"

include("coarse_grain.jl")
include("add_surface_fluxes.jl")
include("animations.jl")
include("training_data.jl")

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
