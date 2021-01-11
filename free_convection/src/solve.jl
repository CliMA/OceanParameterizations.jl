function solve_nde(nde, NN, T₀, alg, nde_params)
    nn_weights, _ = Flux.destructure(NN)
    # See: https://github.com/SciML/DiffEqFlux.jl/blob/449efcecfc11f1eab65d0e467cf57db9f5a5dbec/src/neural_de.jl#L68
    return solve(nde, alg, reltol=1e-4, u0=T₀, p=[nn_weights; nde_params],
                 sense=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))
end

function solve_nde(ds, NN, NDEType, algorithm, T_scaling, wT_scaling; T₀=nothing)
    nde_params = FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    nde = NDEType(NN, ds)

    if isnothing(T₀)
        T₀ = T_scaling.(ds[:T][Ti=1].data)
    end

    T = solve_nde(nde, NN, T₀, algorithm, nde_params) |> Array

    bottom_flux, top_flux, _ = nde_params

    Nz, Nt = size(T)
    wT = zeros(Nz+1, Nt)

    for n in 1:Nt
        wT_interior = NN(T[:, n])
        wT_n = cat(bottom_flux, wT_interior, top_flux, dims=1)
        wT[:, n] .= wT_n
    end

    return (T=T, wT=wT)
end
