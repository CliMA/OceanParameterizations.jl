using Statistics
using Oceananigans.Grids

"""
    coarse_grain(Φ, n)

Average or coarse grain a cell-centered field `Φ` down to a size `n`. `Φ` is required to have evenly spaced points and n <= length(Φ).
"""
function coarse_grain(Φ, n, ::Type{Cell})
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

function coarse_grain(Φ, n, ::Type{Face})
    N = length(Φ)
    Δ = Int((N-2) / (n-2))
    Φ̅ = similar(Φ, n)
    Φ̅[1], Φ̅[n] = Φ[1], Φ[N]
    Φ̅[2:n-1] .= coarse_grain(Φ[2:N-1], n-2, Cell)
    return Φ̅
end
