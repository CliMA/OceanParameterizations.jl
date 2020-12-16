function ConvectiveAdjustmentNDE(NN, ds; iterations=nothing)
    weights, reconstruct = Flux.destructure(NN)

    zc = dims(ds[:T], ZDim)
    zf = dims(ds[:wT], ZDim)
    times = dims(ds[:wT], Ti)
    Nz, Nt = length(zc), length(times)

    H = abs(zf[1]) # Domain height
    τ = times[end]  # Simulation length

    Δẑ = diff(zc[:])[1] / H  # Non-dimensional grid spacing
    Dzᶜ = Dᶜ(Nz, Δẑ) # Differentiation matrix operator
    Dzᶠ = Dᶠ(Nz, Δẑ) # Differentiation matrix operator
    Dzᶜ = convert(Array{eltype(ds[:T])}, Dzᶜ)
    Dzᶠ = convert(Array{eltype(ds[:T])}, Dzᶠ)

    if isnothing(iterations)
        iterations = 1:length(times)
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
        ∂z_wT = Dzᶜ * wT

        # Convective adjustment
        ∂T∂z = Dzᶠ * T
        ∂z_K∂T∂z = Dzᶜ * min.(0, 1 * ∂T∂z)

        return σ_wT/σ_T * τ/H * (- ∂z_wT .+ ∂z_K∂T∂z)
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
