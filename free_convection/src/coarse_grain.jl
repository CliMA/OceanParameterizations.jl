function coarse_grain(ϕ::GeoArray, n, ::Cell)
    N = size(ϕ, zC)
    Δ = Int(N / n)
    ψ = similar(ϕ, n)
    for i in 1:n
        ψ[i] = mean(Φ[Δ*(i-1)+1:Δ*i], )
    end
    return Φ̅
end

function coarse_grain(ds, grid_points)
