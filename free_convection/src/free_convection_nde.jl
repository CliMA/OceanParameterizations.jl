function FreeConvectionNDE(NN, ds; iterations=nothing)
    weights, reconstruct = Flux.destructure(NN)
    zc = dims(ds[:T], ZDim)
    zf = dims(ds[:wT], ZDim)
    times = dims(ds[:wT], Ti)
    Nz, Nt = length(zc), length(times)

    H = abs(zf[1]) # Domain height
    τ = times[end]  # Simulation length

    Δẑ = diff(zc[:])[1] / H  # Non-dimensional grid spacing
    Dzᶜ = Dᶜ(Nz, Δẑ) # Differentiation matrix operator

    if isnothing(iterations)
        iterations = 1:length(times)
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
        ∂z_wT = Dzᶜ * σ_wT/σ_T * τ/H * wT
        return -∂z_wT
    end

    FT = eltype(ds[:wT])
    tspan = FT.( (0.0, maximum(iterations) / Nt) )
    saveat = range(tspan[1], tspan[2], length=length(iterations))

    # See: https://github.com/SciML/DiffEqFlux.jl/blob/449efcecfc11f1eab65d0e467cf57db9f5a5dbec/src/neural_de.jl#L66-L67
    # We set the initial condition to `nothing`. We set it to some actual
    # initial condition when calling `solve`.
    ff = ODEFunction{false}(∂T∂t, tgrad=DiffEqFlux.basic_tgrad)
    return ODEProblem{false}(ff, nothing, tspan, saveat=saveat)
end

function FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    zf = dims(ds[:wT], ZDim)
    times = dims(ds[:wT], Ti)

    H = abs(zf[1]) # Domain height
    τ = times[end]  # Simulation length

    bottom_flux = wT_scaling(ds[:wT][zF=1, Ti=1])
    top_flux = wT_scaling(ds.metadata[:heat_flux])

    FT = eltype(ds[:wT])
    return FT.([bottom_flux, top_flux, T_scaling.σ, wT_scaling.σ, H, τ])
end

function solve_free_convection_nde(nde, NN, T₀, alg, nde_params)
    nn_weights, _ = Flux.destructure(NN)
    # See: https://github.com/SciML/DiffEqFlux.jl/blob/449efcecfc11f1eab65d0e467cf57db9f5a5dbec/src/neural_de.jl#L68
    return solve(nde, alg, reltol=1e-3, u0=T₀, p=[nn_weights; nde_params],
                 sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end
