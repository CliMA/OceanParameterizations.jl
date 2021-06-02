function solve_nde(nde, NN, T₀, alg, nde_params)
    nn_weights, _ = Flux.destructure(NN)
    # See: https://github.com/SciML/DiffEqFlux.jl/blob/449efcecfc11f1eab65d0e467cf57db9f5a5dbec/src/neural_de.jl#L68
    return solve(nde, alg, reltol=1e-4, u0=T₀, p=[nn_weights; nde_params],
                 sense=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
end

function solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling; T₀=nothing)

    T = ds["T"]
    wT = ds["wT"]
    Nz = size(T, 3)
    zc = znodes(T)
    zf = znodes(wT)

    H = abs(zf[1]) # Domain depth/height

    Δẑ = diff(zc)[1] / H  # Non-dimensional grid spacing
    Dzᶠ = Dᶠ(Nz, Δẑ) # Differentiation matrix operator

    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    nde = NDEType(NN, ds)

    if isnothing(T₀)
        T₀ = T_scaling.(interior(T)[1, 1, :, 1])
    end

    T = solve_nde(nde, NN, T₀, algorithm, nde_params) |> Array

    bottom_flux, top_flux, _ = nde_params

    Nz, Nt = size(T)
    wT = zeros(Nz+1, Nt)

    for n in 1:Nt
        T_n = T[:, n]

        wT_NN_interior = NN(T_n)
        wT_NN_n = cat(bottom_flux, wT_NN_interior, top_flux, dims=1)

        if NDEType == FreeConvectionNDE
            @. wT[:, n] = wT_NN_n
        elseif NDEType == ConvectiveAdjustmentNDE
            ∂T∂z_n = Dzᶠ * T_n
            K∂T∂z_n = min.(0, 10 * ∂T∂z_n)
            @. wT[:, n] = wT_NN_n - K∂T∂z_n
        end
    end

    return (T=inv(T_scaling).(T), wT=inv(wT_scaling).(wT))
end
