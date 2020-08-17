using DiffEqFlux: FastLayer

"""
    ConservativeFluxLayer{T, B} <: FastLayer

A neural network layer for imposing a specific flux at the top and bottom of a column model for some physical quantity.
"""
struct ConservativeFluxLayer{T, B} <: FastLayer
              N :: Int
       top_flux :: T
    bottom_flux :: B

    function ConservativeFluxLayer(N; top_flux, bottom_flux)
        return new{typeof(top_flux), typeof(bottom_flux)}(N, top_flux, bottom_flux)
    end
end

(L::ConservativeFluxLayer)(ϕ, p) = [L.top_flux, ϕ..., L.bottom_flux]

function neural_pde_architecture(N; top_flux, bottom_flux)
    return FastChain(FastDense(N, 4N),
                     FastDense(4N, N-2),
                     ConservativeFluxLayer(N; top_flux=top_flux, bottom_flux=bottom_flux))
end
