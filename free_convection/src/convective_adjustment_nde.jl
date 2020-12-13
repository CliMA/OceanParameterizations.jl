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

function solve_convective_adjustment_nde(nde, NN, T₀, alg, nde_params)
    nn_weights, _ = Flux.destructure(NN)
    return solve(nde, alg, reltol=1e-3, u0=T₀, p=[nn_weights; nde_params],
                 sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end