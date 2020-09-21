module Layers

export ConservativeDiffusionLayer

using LinearAlgebra

using DiffEqFlux: FastLayer

struct ConservativeDiffusionLayer{C, K} <: FastLayer
    C :: C
    κ :: K

    function ConservativeDiffusionLayer(N, κ)
        # Define conservation matrix
        C = Matrix(1.0I, N, N)
        C[end, 1:end-1] .= -1
        C[end, end] = 0
        return new{typeof(C), typeof(κ)}(C, κ)
    end
end

(L::ConservativeDiffusionLayer)(u, p) = L.κ * L.C * u

end