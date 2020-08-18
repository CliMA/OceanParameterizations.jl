using Statistics

# weights(f::FastDense, θ) = reshape(θ[1:(f.out*f.in)], f.out, f.in)
# bias(f::FastDense, θ) = θ[(f.out*f.in+1):end]

"""
    coarse_grain(Φ, n)

Average or coarse grain a field `Φ` down to a size `n`. `Φ` is required to have evenly spaced points and n <= length(Φ).
"""
function coarse_grain(Φ, n)
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end
