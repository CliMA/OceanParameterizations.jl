"""
    Dᶠ(N, Δ)

Returns a discrete 1D derivative operator for taking the derivative of a cell-centered field with `N` grid points and `Δ` grid spacing and producing a face-centered field with `N+1` grid points.
"""
function Dᶠ(N, Δ)
    D = zeros(N, N+1)
    for k in 1:N
        D[k, k]   = -1.0
        D[k, k+1] =  1.0
    end
    D = 1/Δ * D
    return D
end

"""
    Dᶜ(N, Δ)

Returns a discrete 1D derivative operator for taking the derivative of a face-centered field with `N+1` grid points and `Δ` grid spacing and producing a cell-centered field with `N` grid points.
"""
function Dᶜ(N, Δ)
    D = zeros(N+1, N)
    for k in 2:N
        D[k, k-1] = -1.0
        D[k, k]   =  1.0
    end
    D = 1/Δ * D
    return D
end
