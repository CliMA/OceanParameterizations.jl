using Oceananigans.Grids: Center, Face

"""
    coarse_grain(Φ, n, ::Type{Center})

Average or coarse grain a `Center`-centered field `Φ` down to size `n`. `Φ` is required to have evenly spaced points and `n` needs to evenly divide `length(Φ)`.
"""
function coarse_grain(Φ, n, ::Type{Center})
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

"""
    coarse_grain(Φ, n, ::Type{Face})

Average or coarse grain a `Face`-centered field `Φ` down to size `n`. `Φ` is required to have evenly spaced points. The values at the left and right endpoints of `Φ` will be preserved in the output.
"""
function coarse_grain(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Δ = (N-2) / (n-2)
    Φ̅[1], Φ̅[n] = Φ[1], Φ[N]

    if isinteger(Δ)
        Φ̅[2:n-1] .= coarse_grain(Φ[2:N-1], n-2, Center)
    else
        for i in 2:n-1
            i1 = round(Int, 2 + (i-2)*Δ)
            i2 = round(Int, 2 + (i-1)*Δ)
            Φ̅[i] = mean(Φ[i1:i2])
        end
    end

    return Φ̅
end

"""
    coarse_grain_linear_interpolation(Φ, n, ::Type{Face})

Average or coarse grain a `Face`-centered field `Φ` down to size `n` using linear interpolation. `Φ` is required to have evenly spaced points. The values at the left and right endpoints of `Φ` will be preserved in the output.
"""
function coarse_grain_linear_interpolation(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Φ̅[1] = Φ[1]
    Φ̅[end] = Φ[end]
    gap = (N-1)/(n-1)

    for i=2:n-1
        Φ̅[i] = 1 + (i-1)*gap
    end

    for i=2:n-1
        Φ̅[i] = (floor(Φ̅[i])+1 - Φ̅[i]) * Φ[Int(floor(Φ̅[i]))] + (Φ̅[i] - floor(Φ̅[i])) * Φ[Int(floor(Φ̅[i]))+1]
    end
    return Φ̅
end
